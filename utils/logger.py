# utils/logger.py — Simple console + file logger

import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR, LOG_FILE


def get_logger(name="FL", level=logging.INFO):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger    = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s][%(name)s] %(message)s",
                                datefmt="%H:%M:%S")
        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
