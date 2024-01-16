import json
import pickle
from srcs.dense_layer import DenseLayer


class Layers:
    def __init__(self, layers=None):
        if layers != None:
            self.layers = layers
        else:
            self.layers = []

    def __str__(self):
        return "[ " + "\n".join(str(layer) for layer in self.layers) + " ]"

    def __len__(self):
        return len(self.layers) if hasattr(self, "layers") else 0

    def __getitem__(self, index):
        if isinstance(index, int):
            # Handle negative indices
            if index < 0:
                index += len(self.layers)
            return self.layers[index] if 0 <= index < len(self.layers) else None
        else:
            # Handle slices if needed
            return self.layers[index]

    def DenseLayer(self, *args, **kwargs):
        # Create a DenseLayer and add it to the list
        layer = DenseLayer(*args, **kwargs)
        return layer

    def save_network(self, filepath):
        self.save_layers_to_json(filepath)
        self.save_layer_parameters(filepath)

    def load_network(self, filepath):
        self.load_layers_from_json(filepath)
        name_without_extension = filepath.split(".")[0]
        self.load_layers_params(name_without_extension)

    def save_layers_to_json(self, filepath):
        filepath = filepath + ".json"
        indexed_data = {
            index: entry.layer_info_to_dict() for index, entry in enumerate(self.layers)
        }
        json_data = json.dumps(indexed_data, indent=2)

        with open(filepath, "w") as file:
            file.write(json_data)

        print(f"Network topology saved to {filepath}")

    def load_layers_from_json(self, filepath):
        with open(filepath, "r") as file:
            json_data = json.load(file)

        for key, value in json_data.items():
            self.layers.append(
                DenseLayer(
                    value["shape"], value["activation"], value["weights_initializer"]
                )
            )

    def load_layers_params(self, name):
        for index, layer in enumerate(self.layers):
            layer.load_parameters(f"{name}_params_{index}.npz")

    def save_layer_parameters(self, name):
        for index, layer in enumerate(self.layers):
            layer.save_parameters(f"{name}_params_{index}")
            print(
                f"Layer parameters for layer {index} saved as {name}_params_{index}.npz"
            )
