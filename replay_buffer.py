import random
from collections import deque, namedtuple
import numpy as np

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
    """
    Buffer 1-step, dùng cho vanilla DQN.

    Dùng list + con trỏ vòng thay vì deque để random.sample chạy O(batch_size)
    thay vì O(n × batch_size) như deque (deque indexing là O(n)).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: list = []
        self._pos: int  = 0

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, state, action, reward, next_state, done,
            legal_mask, next_legal_mask):
        t = Transition(state, action, reward, next_state, done,
                       legal_mask, next_legal_mask)
        if len(self._buf) < self.capacity:
            self._buf.append(t)
        else:
            self._buf[self._pos] = t
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self._buf, batch_size)
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

    Main buffer dùng list + con trỏ vòng để random.sample O(batch_size).
    _window vẫn dùng deque vì kích thước nhỏ (tối đa n_steps phần tử).
    """

    def __init__(self, capacity: int, n_steps: int, gamma: float):
        self.capacity = capacity
        self.n_steps  = n_steps
        self.gamma    = gamma
        self._buf: list = []
        self._pos: int  = 0
        self._window = deque()   # rolling window, tối đa n_steps entries

    def __len__(self) -> int:
        return len(self._buf)

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

        t_new = NStepTransition(
            state=state, action=action,
            reward=base[2], next_state=base[3], done=base[4],
            legal_mask=legal_mask, next_legal_mask=base[6],
            n_step_reward=n_reward,
            n_step_next_state=n_next_state,
            n_step_done=float(n_done),
            n_step_next_legal_mask=n_next_legal_mask,
        )

        if len(self._buf) < self.capacity:
            self._buf.append(t_new)
        else:
            self._buf[self._pos] = t_new
        self._pos = (self._pos + 1) % self.capacity

        self._window.popleft()

    def sample(self, batch_size: int) -> NStepTransition:
        batch = random.sample(self._buf, batch_size)
        return NStepTransition(*zip(*batch))


# ─────────────────────────────────────────────────────────────────────────────
# Sum Tree (dùng cho PER)
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary sum tree cho O(log n) priority sampling.

    Leaf nodes lưu priority, internal nodes lưu tổng con.
    Index 1 = root. Leaf i nằm ở index i + capacity.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._data: list = [None] * capacity
        self._pos  = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def total(self) -> float:
        return float(self._tree[1])

    def add(self, priority: float, data) -> None:
        leaf = self._pos + self.capacity
        self._data[self._pos] = data
        self._set(leaf, priority)
        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, leaf: int, priority: float) -> None:
        self._set(leaf, priority)

    def _set(self, leaf: int, priority: float) -> None:
        delta = priority - self._tree[leaf]
        self._tree[leaf] = priority
        idx = leaf >> 1
        while idx >= 1:
            self._tree[idx] += delta
            idx >>= 1

    def get(self, s: float):
        """Trả về (leaf_index, priority, data) tại vị trí s trong phân phối."""
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if s <= self._tree[left]:
                idx = left
            else:
                s -= self._tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, float(self._tree[idx]), self._data[data_idx]


# ─────────────────────────────────────────────────────────────────────────────
# PER + N-step Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class PERNStepReplayBuffer:
    """
    Prioritized Experience Replay + N-step return.

    Tại sao PER tốt hơn uniform sampling:
    - Ưu tiên lấy mẫu những transition có TD-error lớn → học nhiều hơn từ
      các tình huống model chưa nắm vững
    - alpha điều chỉnh mức độ ưu tiên (0 = uniform, 1 = full priority)
    - IS weights bù trừ bias do sampling không đều

    sample() trả về (batch, indices, is_weights) thay vì chỉ batch.
    Sau update, gọi update_priorities(indices, td_errors) để cập nhật cây.
    """

    def __init__(self, capacity: int, n_steps: int, gamma: float,
                 alpha: float = 0.6, per_eps: float = 1e-6):
        self.n_steps  = n_steps
        self.gamma    = gamma
        self.alpha    = alpha
        self.per_eps  = per_eps
        self._tree    = SumTree(capacity)
        self._window  = deque()
        self._max_p   = 1.0   # priority tối đa — dùng cho transition mới

    def __len__(self) -> int:
        return len(self._tree)

    def add(self, state, action, reward, next_state, done,
            legal_mask, next_legal_mask):
        self._window.append((state, action, reward, next_state, done,
                             legal_mask, next_legal_mask))
        if len(self._window) >= self.n_steps:
            self._store()
        if done:
            while self._window:
                self._store()

    def _store(self):
        if not self._window:
            return
        base = self._window[0]
        state, action, _, _, _, legal_mask, _ = base

        n_reward = 0.0
        n_done   = False
        n_next_state      = base[3]
        n_next_legal_mask = base[6]
        for i, t in enumerate(self._window):
            n_reward         += (self.gamma ** i) * t[2]
            n_next_state      = t[3]
            n_next_legal_mask = t[6]
            if t[4]:
                n_done = True
                break

        t_new = NStepTransition(
            state=state, action=action,
            reward=base[2], next_state=base[3], done=base[4],
            legal_mask=legal_mask, next_legal_mask=base[6],
            n_step_reward=n_reward,
            n_step_next_state=n_next_state,
            n_step_done=float(n_done),
            n_step_next_legal_mask=n_next_legal_mask,
        )
        # Transition mới luôn nhận max priority → được sample ít nhất 1 lần
        self._tree.add(self._max_p, t_new)
        self._window.popleft()

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Returns:
            batch:      NStepTransition
            indices:    leaf indices trong SumTree (dùng để update priority sau)
            is_weights: importance sampling weights shape (batch_size,)
        """
        indices, priorities, transitions = [], [], []
        segment = self._tree.total / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self._tree.get(s)
            if data is None:          # an toàn nếu tree chưa đầy
                s = random.uniform(0, self._tree.total)
                idx, p, data = self._tree.get(s)
            indices.append(idx)
            priorities.append(p)
            transitions.append(data)

        n     = len(self._tree)
        probs = np.array(priorities, dtype=np.float64) / (self._tree.total + 1e-10)
        weights = (n * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return NStepTransition(*zip(*transitions)), np.array(indices, dtype=np.int64), weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.per_eps) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._tree.update(int(idx), float(p))
            self._max_p = max(self._max_p, float(p))
