import random
from collections import deque, namedtuple

# ── Vanilla transition (1-step) ───────────────────────────────────────────────
Transition = namedtuple("Transition", (
    "state", "action", "reward", "next_state", "done",
    "legal_mask", "next_legal_mask",
))

# ── N-step transition ─────────────────────────────────────────────────────────
NStepTransition = namedtuple("NStepTransition", (
    "state", "action", "reward", "next_state", "done",
    "legal_mask", "next_legal_mask",
    # N-step extras
    "n_step_reward",          # Σ γ^i * r_{t+i}
    "n_step_next_state",      # s_{t+N}
    "n_step_done",            # episode ended within N steps
    "n_step_next_legal_mask", # legal mask tại s_{t+N}
))


class ReplayBuffer:
    """Buffer 1-step, dùng cho vanilla DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done,
            legal_mask, next_legal_mask):
        self.buffer.append(
            Transition(state, action, reward, next_state, done,
                       legal_mask, next_legal_mask)
        )

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))


class NStepReplayBuffer:
    """
    Buffer với N-step return, dùng cho Double DQN.

    Duy trì một rolling window N bước. Khi window đủ N hoặc episode kết thúc,
    push transition cũ nhất vào buffer chính kèm N-step reward tính sẵn.

    Tại sao N-step tốt hơn 1-step:
    - 1-step target chỉ propagate reward 1 bước → học chậm khi reward thưa
    - N-step target đưa thông tin reward đi xa hơn mỗi update → học nhanh hơn
    - Trade-off: N lớn → bias tăng nếu policy thay đổi nhiều giữa các bước
    """

    def __init__(self, capacity: int, n_steps: int, gamma: float):
        self.buffer  = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma   = gamma
        self._window = deque()   # rolling window, tối đa n_steps entries

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done,
            legal_mask, next_legal_mask):
        self._window.append((state, action, reward, next_state, done,
                             legal_mask, next_legal_mask))

        # Khi window đủ N bước → push transition cũ nhất
        if len(self._window) >= self.n_steps:
            self._store()

        # Cuối episode → flush các transition còn lại trong window
        if done:
            while self._window:
                self._store()

    def _store(self):
        """Tính N-step return cho transition đầu window rồi push vào buffer chính."""
        if not self._window:
            return

        base = self._window[0]
        state, action, _, _, _, legal_mask, _ = base

        # Tích lũy N-step reward, dừng sớm nếu gặp terminal
        n_reward = 0.0
        n_done   = False
        n_next_state      = base[3]  # fallback = 1-step next
        n_next_legal_mask = base[6]

        for i, t in enumerate(self._window):
            n_reward         += (self.gamma ** i) * t[2]  # t.reward
            n_next_state      = t[3]                       # t.next_state
            n_next_legal_mask = t[6]                       # t.next_legal_mask
            if t[4]:  # t.done
                n_done = True
                break

        self.buffer.append(NStepTransition(
            state=state, action=action,
            reward=base[2], next_state=base[3], done=base[4],
            legal_mask=legal_mask, next_legal_mask=base[6],
            n_step_reward=n_reward,
            n_step_next_state=n_next_state,
            n_step_done=float(n_done),
            n_step_next_legal_mask=n_next_legal_mask,
        ))
        self._window.popleft()

    def sample(self, batch_size: int) -> NStepTransition:
        batch = random.sample(self.buffer, batch_size)
        return NStepTransition(*zip(*batch))
