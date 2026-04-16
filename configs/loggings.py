import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Tạo logger ghi ra cả console (INFO) và file (DEBUG).

    Args:
        name (str): Tên logger, dùng làm prefix trong log file.
        log_dir (str): Thư mục lưu file log. Mặc định "logs/".

    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # tránh duplicate nếu gọi lại
        return logger
    logger.setLevel(logging.DEBUG)

    fmt_console = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%H:%M:%S")
    fmt_file    = logging.Formatter("[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s")

    # Console — chỉ INFO trở lên
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    # File — ghi hết DEBUG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
