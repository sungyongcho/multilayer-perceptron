from srcs.dense_layer import DenseLayer


class Layers:
    def __init__(self, layers=None):
        if layers != None:
            self.layers = layers

    def __str__(self):
        return "[ " + "\n".join(str(layer) for layer in self.layers) + " ]"

    def __len__(self):
        return len(self.layers) if hasattr(self, "layers") else 0

    def __getitem__(self, index):
        return self.layers[index] if 0 <= index < len(self.layers) else None

    def DenseLayer(self, *args, **kwargs):
        # Create a DenseLayer and add it to the list
        layer = DenseLayer(*args, **kwargs)
        return layer
