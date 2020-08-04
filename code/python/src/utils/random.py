import numpy as np
import tensorflow as tf


def set_random_seed(seed: int):
    """
    Sets the NumPy and Tensorflow random seeds to the provided value

    :param seed: the random seed value to use
    """
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
