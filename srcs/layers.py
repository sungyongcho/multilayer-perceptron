from srcs.dense_layer import DenseLayer


class Layers:
    def __init__(self, layers=None):
        if layers != None:
            self.layers = layers

    def __str__(self):
        return "[ " + "\n".join(str(layer) for layer in self.layers) + " ]"

    def DenseLayer(self, *args, **kwargs):
        # Create a DenseLayer and add it to the list
        layer = DenseLayer(*args, **kwargs)
        return layer
