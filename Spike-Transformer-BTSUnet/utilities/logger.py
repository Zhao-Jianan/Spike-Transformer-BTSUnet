import logging
import sys
from pathlib import Path


# ---------------------------
# 全局 logger 配置
# ---------------------------
def _init_logger(name=__name__, level=logging.INFO, log_file: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复打印

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



# ---------------------------
# 实例化全局 logger
# ---------------------------

logger = _init_logger(__name__, level=logging.INFO, log_file=None)
