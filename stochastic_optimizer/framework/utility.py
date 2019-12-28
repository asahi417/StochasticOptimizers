import logging
import os


def create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler for logger file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter(
        "H1, %(asctime)s %(levelname)8s %(message)s"))
    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter(
        "H1, %(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
