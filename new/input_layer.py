import tensorflow as tf
import numpy as np


# Custom input layer class
class InputLayer:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

    def build(self):
        # No trainable weights for the input layer
        pass

    def forward(self, inputs):
        return inputs
