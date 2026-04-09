# -*- coding: utf-8 -*-
"""训练脚本通用日志：按 --log_dir 与 --save_name 写文件并打屏。"""

import logging
import os


def setup_train_logging(log_dir=None, save_name="train.pt"):
    """
    若 log_dir 设置，则写入 log_dir/<save_name  stem>.log，并同时输出到控制台。
    返回 logger。
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_stem = os.path.splitext(os.path.basename(save_name))[0] + ".log"
        log_path = os.path.join(log_dir, log_stem)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
    return logging.getLogger(__name__)
