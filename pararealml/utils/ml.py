import tensorflow as tf
from mpi4py import MPI


def use_cpu():
    """
    Ensures that Tensorflow does not use any GPUs.
    """
    tf.config.experimental.set_visible_devices([], 'GPU')


def limit_visible_gpus():
    """
    If there are GPUs available, it sets the GPU corresponding to the MPI rank
    of the process as the only device visible to Tensorflow.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        comm = MPI.COMM_WORLD
        if len(gpus) != comm.size:
            raise ValueError(
                f'number of GPUs ({len(gpus)}) must match default '
                f'communicator size ({comm.size})')
        tf.config.experimental.set_visible_devices(gpus[comm.rank], 'GPU')
