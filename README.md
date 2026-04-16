# DQN-2048

Train agent chơi 2048 bằng Double DQN + Dueling Network, dùng [OpenSpiel](https://github.com/google-deepmind/open_spiel) làm game engine.

---

## Kiến trúc

### Mạng Dueling DQN (`network.type: "dueling"`)

```
obs (288 floats)
  │
  │  OpenSpiel encode board 4×4 thành one-hot:
  │  16 ô × 18 channel = 288 floats
  │  channel k = 1.0 nếu ô đó chứa tile 2^k (k=0 → trống)
  │
  ▼ reshape (4,4,18) → permute → (18, 4, 4)
  │
  ▼ Conv2d(18→64,  k=3, pad=1) + ReLU   → (64,  4, 4)
  ▼ Conv2d(64→128, k=3, pad=1) + ReLU   → (128, 4, 4)
  ▼ Conv2d(128→128, k=1)       + ReLU   → (128, 4, 4)
  ▼ Flatten                              → 2048 features
  │
  ▼ Linear(2048→256) + LayerNorm + ReLU
  ▼ Linear(256→256)  + ReLU
  │
  ├─▶ Value head:     256→128→1          V(s)
  └─▶ Advantage head: 256→128→4          A(s,a)
  │
  ▼ Q(s,a) = V(s) + A(s,a) − mean_a A(s,.)
```

> **Lưu ý quan trọng:** CNN nhận đầu vào là **18 kênh** (toàn bộ one-hot encoding), không phải 16 float đầu tiên. Mỗi kênh là bản đồ 4×4 cho một giá trị tile cụ thể — CNN học được pattern không gian của từng loại tile trên board.

### Các kiến trúc khác

| `network.type` | Mô tả |
|----------------|-------|
| `vanilla` | MLP 2 lớp ẩn |
| `deep` | MLP 3 lớp + LayerNorm |
| `dueling` | CNN 18-channel + Dueling heads (khuyến nghị) |

### Thuật toán

- **Double DQN**: q_net chọn action, target_net đánh giá → giảm overestimation bias
- **N-step return**: tích lũy reward N bước trước khi bootstrap → học nhanh hơn với reward thưa
- **Epsilon-greedy**: khám phá tuyến tính từ 1.0 → 0.05

---

## Cài đặt

```bash
# Colab
!apt-get install -y build-essential cmake
!pip install open-spiel torch matplotlib tqdm imageio pyyaml numpy

# Local
pip install -r requirements.txt
```

---

## Cách dùng

### 1. Train

```bash
python main.py train
python main.py train --episodes 5000
```

### 2. Hyperparameter search

```bash
# Random search rộng (khám phá ban đầu)
python main.py search --mode random --trials 10 --episodes 500

# Thu hẹp vào vùng tốt (sau khi có kết quả broad search)
python main.py search --space narrow --mode grid --episodes 1500

# So sánh kiến trúc mạng (vanilla vs deep vs dueling, cùng params)
python main.py search --space net --mode grid --episodes 1500
```

| `--space` | Mô tả |
|-----------|-------|
| `wide` | Toàn bộ search space (mặc định) |
| `narrow` | Thu hẹp từ kết quả broad search: dueling, γ=0.99, hid=256, bs=128 |
| `net` | So sánh kiến trúc: fix params tốt nhất, vary network_type |

### 3. Inference (chạy model đã train)

```bash
# Từ file checkpoint cụ thể
python main.py inference checkpoints/dqn_ep02000_step123456.pt

# Từ thư mục (tự load best_model.pt)
python main.py inference checkpoints/

# Thêm options
python main.py inference checkpoints/ --episodes 20 --render
```

`--render` in board từng bước ra terminal.

### 4. Visualize kết quả search

```bash
python main.py visualize
```

---

## Config (`config.yaml`)

```yaml
training:
  gamma: 0.99                  # discount factor
  learning_rate: 5e-4          # Adam lr
  batch_size: 128
  replay_capacity: 100000
  learn_start: 1000            # bắt đầu update sau n transition
  learn_every: 4               # update mỗi n bước
  target_sync_every: 500       # sync target net mỗi n bước
  num_episodes: 2000
  max_steps_per_episode: 10000
  use_double_dqn: true         # false → vanilla DQN
  n_steps: 3                   # N-step return (khi use_double_dqn=true)
  loss: "huber"                # mse | huber | smooth_l1

epsilon:
  start: 1.0
  end: 0.05
  decay_steps: 100000          # scale theo tổng số step khi train dài

network:
  type: "dueling"              # vanilla | deep | dueling
  hidden_dim: 256

env:
  seed: 42
```

### Scale config cho train dài

| Epochs | `decay_steps` | `replay_capacity` | `target_sync_every` |
|--------|---------------|-------------------|---------------------|
| 800 | 50,000 | 50,000 | 500 |
| 3,000 | 150,000 | 100,000 | 500 |
| 20,000 | 500,000 | 200,000 | 1,000 |

> `decay_steps` phải scale tỉ lệ với số episode — nếu để nguyên, epsilon về 0.05 quá sớm, agent không explore đủ.

---

## Kết quả thực nghiệm

### Round 1 — Broad search, 10 trials, 500 ep/trial

| Rank | lr | γ | N | net | hid | eps_steps | MeanEval | BestEval |
|------|----|---|---|-----|-----|-----------|----------|----------|
| 1 | 1e-3 | 0.99 | 3 | dueling | 256 | 50k | 2546.9 | 3418.7 |
| 2 | 3e-4 | 0.99 | 5 | dueling | 256 | 100k | 2149.1 | 4500.0 |
| 3 | 1e-4 | 0.95 | 1 | dueling | 256 | 50k | 2043.1 | 3694.7 |

**Kết luận round 1:** dueling > deep > vanilla, γ=0.99, hid=256, bs=128.

---

### Round 2 — Net compare, 3 trials, 1500 ep/trial (cùng params)

| net | MeanEval | BestEval |
|-----|----------|----------|
| dueling | 2391.0 | 4370.7 |
| deep | 1955.0 | 4174.7 |
| vanilla | 1817.4 | 4394.7 |

**Kết luận round 2:** Dueling nhất quán tốt hơn. Confirmed.

---

### Round 3 — Narrow search, 10 trials, 1500 ep/trial

Fix all: γ=0.99, net=dueling, hid=256, bs=128. Vary: lr × N × eps_steps.

| Rank | lr | N | eps_steps | MeanEval | BestEval |
|------|----|---|-----------|----------|----------|
| 1 | **3e-4** | 3 | **50k** | **2920.9** | 4261.3 |
| 2 | 1e-3 | 3 | 50k | 2720.0 | 5366.7 |
| 3 | 5e-4 | 3 | 100k | 2706.8 | 4397.3 |
| 4 | 1e-3 | 5 | 50k | 2661.2 | 4732.0 |
| 5 | 3e-4 | 5 | 100k | 2415.1 | 4370.7 |
| 6 | 5e-4 | 5 | 100k | 2205.7 | 4025.3 |
| 7 | 1e-3 | 3 | 100k | 2074.0 | **5665.3** |
| 8 | 3e-4 | 3 | 100k | 1838.8 | 3062.7 |
| 9 | 5e-4 | 3 | 50k | 1640.6 | 3824.0 |
| 10 | 1e-3 | 5 | 100k | 1593.5 | 3200.0 |

#### Phân tích Round 3

**Learning rate:**

| lr | MeanEval trung bình (tất cả trials) |
|----|-------------------------------------|
| 3e-4 | (2920.9 + 2415.1 + 1838.8) / 3 = **2391** — ổn định nhất |
| 1e-3 | (2720.0 + 2661.2 + 2074.0 + 1593.5) / 4 = 2262 — variance cao |
| 5e-4 | (2706.8 + 2205.7 + 1640.6) / 3 = 2184 — yếu nhất |

→ **lr=3e-4 ổn định nhất**, lr=1e-3 có thể đỉnh cao (BestEval=5665) nhưng không nhất quán.

**N-step:**
- N=3 avg: 2316 | N=5 avg: 2219 → **N=3 tốt hơn** cho 1500 ep.

**eps_steps:**
- eps_steps=50k avg: **2485** | eps_steps=100k avg: 2139
- **eps_steps=50k thắng rõ** ở 1500 ep vì agent exploit sớm hơn trong budget giới hạn.

**BestEval vs MeanEval:**
- trial_007 (lr=1e-3, eps=100k): MeanEval=2074 nhưng BestEval=5665 → **variance cao, không ổn định**.
- trial_010 (lr=3e-4, eps=50k): MeanEval=2920, BestEval=4261 → **ổn định, đáng tin hơn**.
- Ưu tiên MeanEval cao khi chọn config cho train dài.

#### Best config (từ narrow search)

```
lr = 3e-4   |  γ = 0.99  |  N = 3
eps_steps = 50k (search)  →  scale lên 500k khi train 20k ep
bs = 128    |  net = dueling  |  hid = 256
```

---

## Workflow khuyến nghị

```
1. Broad search  →  python main.py search --episodes 500 --trials 10
                    (tìm vùng param tốt)

2. Net compare   →  python main.py search --space net --mode grid --episodes 1500
                    (xác nhận kiến trúc)

3. Narrow search →  python main.py search --space narrow --mode grid --episodes 1500
                    (fine-tune lr, n_steps, eps_steps)

4. Copy best config và scale:
                    cp grid_search_results/best_config.yaml config.yaml
                    # chỉnh decay_steps theo bảng Scale ở trên

5. Train thật    →  python main.py train --episodes 10000

6. Đánh giá      →  python main.py inference checkpoints/ --episodes 50
```

---

## Cấu trúc project

```
DQN-BT2/
├── main.py                  # CLI entry point
├── config.yaml              # config chính
├── config.py                # dataclasses cho config
├── networks.py              # DQNNetwork, DeepDQNNetwork, DuelingDQNNetwork
├── dqn_update.py            # dqn_update(), double_dqn_update()
├── replay_buffer.py         # ReplayBuffer, NStepReplayBuffer (circular list)
├── environment_game.py      # OpenSpiel 2048 wrapper
├── helper.py                # extract_obs, legal_actions, ...
├── training/
│   ├── train.py             # training loop
│   ├── load_models.py       # khởi tạo mọi component từ config
│   └── grid_search.py       # hyperparameter search
├── inference/
│   └── inference.py         # chạy model đã train
├── models/
│   ├── save_model.py
│   └── load_model.py
├── visualize/
│   └── visualize.py         # plot training curves, board
├── utils/
│   └── losses.py            # compute_loss()
└── configs/
    └── loggings.py          # get_logger()
```
