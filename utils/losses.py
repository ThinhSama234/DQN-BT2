"""
Tập hợp các loss function dùng cho DQN training.
Import và dùng qua: from utils.losses import compute_loss
"""
import torch
import torch.nn.functional as F


_REGISTRY: dict[str, callable] = {
    "mse":       F.mse_loss,
    "huber":     F.huber_loss,       # delta=1 mặc định, ít nhạy cảm với outlier hơn MSE
    "smooth_l1": F.smooth_l1_loss,   # tương đương huber với delta=1
}


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    """
    Tính loss giữa Q-value dự đoán và target.

    Args:
        pred:      Q(s,a) — shape (batch,)
        target:    Bellman target — shape (batch,)
        loss_type: "mse" | "huber" | "smooth_l1"

    Returns:
        Scalar tensor.

    So sánh:
        mse       → phạt nặng outlier (large TD-error), học nhanh ban đầu
                    nhưng dễ bất ổn khi reward lớn như 2048
        huber     → quadratic khi |err|<=1, linear khi lớn hơn
                    → ổn định hơn với reward lớn, khuyến nghị cho 2048
        smooth_l1 → giống huber, thường dùng trong object detection/DQN Atari
    """
    fn = _REGISTRY.get(loss_type)
    if fn is None:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Available: {list(_REGISTRY)}")
    return fn(pred, target)


def available_losses() -> list[str]:
    return list(_REGISTRY)
