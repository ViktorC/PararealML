from abc import abstractmethod

import numpy as np
import tensorflow as tf

from pararealml.core.operators.pidon.collocation_point_sampler import CollocationPointSet


class PIDeepONet:

    def __init__(self):
        ...

    def train(self, collocation_points: CollocationPointSet, epochs: int):
        ...
