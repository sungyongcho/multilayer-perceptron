from DenseLayer import DenseLayer


class Layers:
    def __init__(self):
        pass

    def DenseLayer(self, *args, **kwargs):
        # Create a DenseLayer and add it to the list
        layer = DenseLayer(*args, **kwargs)
        return layer
