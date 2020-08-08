import tensorflow as tf
from mpi4py import MPI

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    comm = MPI.COMM_WORLD
    assert len(gpus) == comm.size
    tf.config.experimental.set_visible_devices(gpus[comm.rank], 'GPU')
