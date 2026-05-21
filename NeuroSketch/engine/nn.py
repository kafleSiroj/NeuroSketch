import numpy as np
from ._module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, init_type=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = 0.01
        if init_type=="he":
            std = np.sqrt(2 / in_features)
        elif init_type=="xavier":
            std = np.sqrt(2 / (in_features + out_features))
        elif init_type=="zero":
            std = 0

        self.weight = np.random.randn(out_features, in_features) * std
        self.bias = np.zeros((out_features, 1))
        self._register_params(self.weight, self.bias)

    def forward(self, x):
        val = x @ self.weight.T + self.bias.T
        self.input = x 
        self.out = val
        return val

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features})"


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_params(self):
        all_params = {"weights": [], "biases": []}

        for layer in self.layers:
            if hasattr(layer, "params") and layer.params:
                all_params["weights"].extend(layer.params.get("weights", []))
                all_params["biases"].extend(layer.params.get("biases", []))

        return all_params
    
    def summary(self):
        lines = []

        lines.append("=" * 70)
        lines.append(f"{'Idx':<5}{'Layer':<45}{'Params':>15}")
        lines.append("=" * 70)

        total_params = 0

        for i, layer in enumerate(self.layers):

            params = 0

            if hasattr(layer, "weight"):
                params += layer.weight.size             

            if hasattr(layer, "bias"):
                params += layer.bias.size

            total_params += params

            lines.append(
                f"{i:<5}"
                f"{repr(layer):<45}"
                f"{params:>15,}"
            )

        lines.append("=" * 70)
        lines.append(f"{'Total Parameters':<50}{total_params:>20,}")
        lines.append("=" * 70)

        return "\n".join(lines)