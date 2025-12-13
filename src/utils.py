import logging
import os
import sys
from typing import Optional


def setup_logger(
    *,
    name: str = "run",
    log_dir: str,
    filename: str = "run.log",
    level: int = logging.INFO,
    mode: str = "w",
) -> logging.Logger:
    """Configure a file+stdout logger.

    Intended for scripts under src/ so Docker captures stdout while a full log is
    also persisted under log/.
    """

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    # Reset handlers (important for notebooks / re-entrancy)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode=mode),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    logger.info(f"Logging initialized. Saving logs to: {log_path}")
    return logger
