import logging
import os
import numpy as np

import sys


def load_path():
    # Get the path of the current script's directory
    current_dir = os.path.dirname(os.path.realpath("__file__"))
    # Get the path of the parent directory
    parent_dir = os.path.dirname(current_dir)
    # Add the parent directory to sys.path
    sys.path.append(parent_dir)


class LogUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        match os.getenv("LOG_LEVEL"):
            case "INFO":
                logging.basicConfig(level=logging.INFO)
            case "DEBUG":
                logging.basicConfig(level=logging.DEBUG)
            case "ERROR":
                logging.basicConfig(level=logging.ERROR)
            case _:
                logging.basicConfig(level=logging.INFO)

    def get_logging(self):
        return self.logger


def cosine_sim(a: np.array, b: np.array):
    """
    Compute cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: cosine similarity.
    """
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b) if b.ndim == 1 else np.linalg.norm(b, axis=0)
    sim = np.dot(a, b) / (a_norm * b_norm)
    return sim
