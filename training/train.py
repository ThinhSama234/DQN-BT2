import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch

from training.load_models import (
    cfg, train_env, q_net, target_net, optimizer, scheduler, replay,
    obs_dim, num_actions, DEVICE,
    SEED, NUM_EPISODES, MAX_STEPS_PER_EPISODE,
    LEARN_START, LEARN_EVERY, BATCH_SIZE, TARGET_SYNC_EVERY,
    epsilon_by_step, guide_prob_by_step, dqn_update, per_beta_by_step,
    make_legal_mask, masked_greedy_action, guide,
    main as log_setup,
)
from replay_buffer import PERNStepReplayBuffer
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

SAVE_EVERY = cfg.training.save_every  # từ config.yaml


def train(resume_from: str = None, output_dir: str = "checkpoints"):
    global global_step, best_eval_return

    # ── Resume từ checkpoint ──────────────────────────────────────────────────
    start_episode = 1
    if resume_from:
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        q_net.load_state_dict(ckpt["q_net_state_dict"])
        target_net.load_state_dict(ckpt["q_net_state_dict"])

        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        global_step      = ckpt.get("global_step", 0)
        start_episode    = ckpt.get("episode", 0) + 1
        best_eval_return = ckpt.get("best_eval_return", float("-inf"))

        # Weights-only checkpoint (best_model.pt cũ): không có step info
        # → set global_step = decay_steps để epsilon bắt đầu ở mức tối thiểu
        weights_only = ckpt.get("optimizer_state_dict") is None
        if weights_only and global_step == 0:
            global_step = cfg.epsilon.decay_steps
            tqdm.write(f"Weights-only: global_step → {global_step} | eps={cfg.epsilon.end:.3f}")
        else:
            tqdm.write(f"Resumed ep={start_episode-1} | step={global_step} | best={best_eval_return:.1f}")

        # Clamp LR về lr_min nếu scheduler cũ đã decay quá thấp
        lr_min = cfg.training.lr_min
        for pg in optimizer.param_groups:
            if pg["lr"] < lr_min:
                pg["lr"] = lr_min
                tqdm.write(f"LR clamped → {lr_min:.1e} (lr_min floor)")
        logger.info("Resumed from %s | ep=%d step=%d best=%.1f",
                    resume_from, start_episode - 1, global_step, best_eval_return)

    recent_loss = 0.0

    # Cache 1 lần — tránh isinstance check trong hot loop
    use_per  = isinstance(replay, PERNStepReplayBuffer)
    use_guide = guide is not None

    # Tạo eval_env một lần, reuse bằng reset() — tránh load pyspiel mỗi lần eval
    eval_env = OpenSpiel2048Env(seed=9999)

    pbar = tqdm(range(start_episode, NUM_EPISODES + 1),
                desc=f"Train ep{start_episode}→{NUM_EPISODES}",
                dynamic_ncols=True)
    for episode in pbar:
        obs  = train_env.reset(seed=SEED + episode)
        done = False
        ep_return = 0.0
        ep_len    = 0

        # Tính 1 lần/episode — eps và g_prob thay đổi rất ít trong 1 episode
        eps    = epsilon_by_step(global_step)
        g_prob = guide_prob_by_step(global_step) if use_guide else 0.0

        while not done and ep_len < MAX_STEPS_PER_EPISODE:
            legal = train_env.legal_actions()
            if not legal:
                break
            legal_mask = make_legal_mask(num_actions, legal)

            # ExpectiMax guide trong giai đoạn đầu (cold start)
            if use_guide and g_prob > 0.0 and random.random() < g_prob:
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
                if use_per:
                    beta  = per_beta_by_step(global_step)
                    batch, per_idx, is_w = replay.sample(BATCH_SIZE, beta)
                    recent_loss, td_err  = dqn_update(batch, is_weights=is_w)
                    replay.update_priorities(per_idx, td_err)
                else:
                    batch = replay.sample(BATCH_SIZE)
                    recent_loss, _  = dqn_update(batch)
                loss_history.append(recent_loss)

            if global_step % TARGET_SYNC_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

        # ── Cập nhật thanh progress sau mỗi episode ───────────────────────────
        pbar.set_postfix(
            ret=f"{ep_return:.0f}",
            best=f"{best_eval_return:.0f}",
            eps=f"{eps:.3f}",
            guide=f"{g_prob:.2f}",
            loss=f"{recent_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
            step=global_step,
        )

        # ── Periodic save ─────────────────────────────────────────────────────
        if episode % SAVE_EVERY == 0:
            path = save_checkpoint(q_net, optimizer, episode, global_step, cfg,
                                   scheduler=scheduler,
                                   best_eval_return=best_eval_return,
                                   save_dir=output_dir)
            logger.info("Checkpoint saved → %s", path)

        # ── LR scheduler step mỗi episode (chỉ sau khi optimizer đã chạy) ───
        if len(replay) >= LEARN_START:
            scheduler.step()

        # ── Evaluation mỗi 20 episode ─────────────────────────────────────────
        if episode % cfg.training.eval_every == 0:
            q_net.eval()
            game_returns = []
            for g in range(cfg.training.eval_games):
                obs_eval   = eval_env.reset(seed=2000 + episode + g)
                done_eval  = False
                ret_g      = 0.0
                steps_eval = 0
                while not done_eval and steps_eval < MAX_STEPS_PER_EPISODE:
                    legal = eval_env.legal_actions()
                    if not legal:
                        break
                    action = masked_greedy_action(q_net, obs_eval, legal, num_actions, epsilon=0.0, device=DEVICE)
                    obs_eval, reward, done_eval, _ = eval_env.step(action)
                    ret_g      += reward
                    steps_eval += 1
                game_returns.append(ret_g)

            ret_eval = sum(game_returns) / len(game_returns)
            eval_returns.append((episode, ret_eval))

            # Lưu best model
            if ret_eval > best_eval_return:
                best_eval_return = ret_eval
                save_best(q_net, optimizer, episode, global_step, cfg,
                          scheduler=scheduler,
                          best_eval_return=best_eval_return,
                          save_dir=output_dir)
                logger.info("New best eval return=%.1f at ep=%d → best_model.pt saved", ret_eval, episode)

            q_net.train()
            current_lr = scheduler.get_last_lr()[0]
            msg = (f"[Eval ep {episode:>5}] avg={ret_eval:.1f} ({cfg.training.eval_games}g)"
                   f" | eps={eps:.3f}"
                   f" | guide={g_prob:.2f}"
                   f" | lr={current_lr:.2e}"
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
