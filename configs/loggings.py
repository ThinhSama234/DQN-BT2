import logging
import os
import sys
from datetime import datetime


class _FlushStreamHandler(logging.StreamHandler):
    """StreamHandler flush ngay sau mỗi emit — cần thiết cho Kaggle/Colab notebook."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt_console = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%H:%M:%S")
    fmt_file    = logging.Formatter("[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s")

    # Console — flush ngay, ghi ra stdout (tqdm dùng stderr nên không conflict)
    ch = _FlushStreamHandler(sys.stdout)
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
