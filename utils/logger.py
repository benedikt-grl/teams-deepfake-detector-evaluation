import logging
import sys


def setup_basic_logger(name):
    # Check if the logger already has handlers to avoid duplication
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Taken from https://stackoverflow.com/questions/7621897/python-logging-module-globally
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger, otherwise messages get logged twice.
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger
