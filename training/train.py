import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch

from training.load_models import (
    cfg, train_env, q_net, target_net, optimizer, replay,
    obs_dim, num_actions, DEVICE,
    SEED, NUM_EPISODES, MAX_STEPS_PER_EPISODE,
    LEARN_START, LEARN_EVERY, BATCH_SIZE, TARGET_SYNC_EVERY,
    epsilon_by_step, guide_prob_by_step, dqn_update,
    make_legal_mask, masked_greedy_action, guide,
    main as log_setup,
)
from environment_game import OpenSpiel2048Env
from configs.loggings import get_logger
from models.save_model import save_checkpoint, save_best
from visualize.visualize import plot_training, plot_eval
from tqdm import tqdm

logger = get_logger("dqn_train", log_dir="logs")

# ── Tracking ──────────────────────────────────────────────────────────────────
episode_returns = []
episode_lengths = []
loss_history    = []
eval_returns    = []

global_step = 0
best_eval_return = float("-inf")

SAVE_EVERY = 200   # lưu checkpoint định kỳ mỗi n episode


def train(resume_from: str = None, output_dir: str = "checkpoints"):
    global global_step, best_eval_return

    # ── Resume từ checkpoint ──────────────────────────────────────────────────
    start_episode = 1
    if resume_from:
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        q_net.load_state_dict(ckpt["q_net_state_dict"])
        target_net.load_state_dict(ckpt["q_net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step   = ckpt.get("global_step", 0)
        start_episode = ckpt.get("episode", 0) + 1
        tqdm.write(f"Resumed from checkpoint: ep={start_episode-1}, step={global_step}")
        logger.info("Resumed from %s | ep=%d step=%d", resume_from, start_episode-1, global_step)

    recent_loss = 0.0

    pbar = tqdm(range(start_episode, NUM_EPISODES + 1), desc="Training", dynamic_ncols=True)
    for episode in pbar:
        obs  = train_env.reset(seed=SEED + episode)
        done = False
        ep_return = 0.0
        ep_len    = 0

        while not done and ep_len < MAX_STEPS_PER_EPISODE:
            eps   = epsilon_by_step(global_step)
            legal = train_env.legal_actions()
            if not legal:
                break
            legal_mask = make_legal_mask(num_actions, legal)

            # ExpectiMax guide trong giai đoạn đầu (cold start)
            g_prob = guide_prob_by_step(global_step)
            if guide is not None and random.random() < g_prob:
                action = guide.best_action(train_env.state, legal)
            else:
                action = masked_greedy_action(
                    qnet=q_net, obs=obs,
                    legal_actions_list=legal, num_actions=num_actions,
                    epsilon=eps, device=DEVICE,
                )

            next_obs, reward, done, info = train_env.step(action)
            next_legal      = info["legal_actions"] if not done else []
            next_legal_mask = make_legal_mask(num_actions, next_legal)

            replay.add(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)

            obs        = next_obs
            ep_return += reward
            ep_len    += 1
            global_step += 1

            if len(replay) >= LEARN_START and global_step % LEARN_EVERY == 0:
                batch = replay.sample(BATCH_SIZE)
                recent_loss = dqn_update(batch)
                loss_history.append(recent_loss)
                logger.debug("step=%d  loss=%.4f", global_step, recent_loss)

            if global_step % TARGET_SYNC_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
                logger.debug("Target net synced at step %d", global_step)

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        logger.debug("ep=%d  return=%.1f  len=%d  eps=%.3f",
                     episode, ep_return, ep_len, epsilon_by_step(global_step))

        # ── Cập nhật thanh progress sau mỗi episode ───────────────────────────
        g_prob = guide_prob_by_step(global_step)
        pbar.set_postfix(
            ret=f"{ep_return:.0f}",
            best=f"{best_eval_return:.0f}",
            eps=f"{epsilon_by_step(global_step):.3f}",
            guide=f"{g_prob:.2f}",
            loss=f"{recent_loss:.4f}",
            step=global_step,
        )

        # ── Periodic save ─────────────────────────────────────────────────────
        if episode % SAVE_EVERY == 0:
            path = save_checkpoint(q_net, optimizer, episode, global_step, cfg,
                                   save_dir=output_dir)
            logger.info("Checkpoint saved → %s", path)

        # ── Evaluation mỗi 20 episode ─────────────────────────────────────────
        if episode % 20 == 0:
            eval_env  = OpenSpiel2048Env(seed=1000 + episode)
            obs_eval  = eval_env.reset(seed=2000 + episode)
            done_eval = False
            ret_eval  = 0.0
            steps_eval = 0

            while not done_eval and steps_eval < MAX_STEPS_PER_EPISODE:
                legal  = eval_env.legal_actions()
                if not legal:
                    break
                action = masked_greedy_action(q_net, obs_eval, legal, num_actions, epsilon=0.0, device=DEVICE)
                obs_eval, reward, done_eval, _ = eval_env.step(action)
                ret_eval   += reward
                steps_eval += 1

            eval_returns.append((episode, ret_eval))

            # Lưu best model
            if ret_eval > best_eval_return:
                best_eval_return = ret_eval
                save_best(q_net, save_dir=output_dir)
                logger.info("New best eval return=%.1f at ep=%d → best_model.pt saved", ret_eval, episode)

            msg = (f"[Eval ep {episode:>5}] return={ret_eval:.1f}"
                   f" | eps={epsilon_by_step(global_step):.3f}"
                   f" | best={best_eval_return:.1f}")
            tqdm.write(msg)
            logger.info(msg)

    logger.info("Training complete. Total steps: %d", global_step)
    print("Training complete.")


def main(resume_from: str = None, output_dir: str = "checkpoints"):
    log_setup()
    if resume_from:
        logger.info("Resuming training from %s | output_dir=%s", resume_from, output_dir)
    else:
        logger.info("Starting training | device=%s | episodes=%d | output_dir=%s",
                    DEVICE, NUM_EPISODES, output_dir)

    train(resume_from=resume_from, output_dir=output_dir)

    # ── Plots sau khi train xong ──────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    plot_training(episode_returns, episode_lengths, loss_history, save_path="plots/training.png")
    plot_eval(eval_returns, save_path="plots/eval.png")
    logger.info("Plots saved to plots/")


if __name__ == "__main__":
    main()
