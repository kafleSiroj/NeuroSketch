# NeuroSketch

A lightweight supervised learning framework built from scratch with NumPy. No black boxes.

Implements fundamental ML concepts: linear layers, activations, loss functions, and optimizers with explicit gradient computation and backpropagation.

## Features

### Layers
- **Linear** - Fully connected layers with He/Xavier initialization

### Activations
- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Swish, HeavySide

### Loss Functions
- MSELoss, MAELoss, BinaryCrossentropyLoss, SparseCategoricalCrossentropyLoss

### Optimizers
- Gradient Descent, SGD, Adam

## Installation

```bash
pip install neurosketch
```

Or from source:
```bash
git clone <repo>
cd neurosketch
pip install -e .
```

## Quick Start

### Binary Classification

```python
import numpy as np
from neurosketch.layers import Linear
from neurosketch.activations import Sigmoid
from neurosketch.losses import BinaryCrossentropyLoss
from neurosketch.optimizers import Adam

# Create model
linear = Linear(in_features=10, out_features=1, init_type="he")
activation = Sigmoid()
criterion = BinaryCrossentropyLoss(linear)

# Forward pass
x = np.random.randn(32, 10)
z = linear.forward(x)
pred = activation.forward(z)
loss = criterion(pred, y)

# Backward pass
criterion.backward()

# Optimize
optimizer = Adam(layers=[linear])
optimizer.step(learning_rate=0.01)
```

### Multiclass Classification

```python
from neurosketch.layers import Linear
from neurosketch.activations import Softmax
from neurosketch.losses import SparseCategoricalCrossentropyLoss

linear = Linear(in_features=10, out_features=3, init_type="xavier")
activation = Softmax()
criterion = SparseCategoricalCrossentropyLoss(linear)

z = linear.forward(x)
pred = activation.forward(z)
loss = criterion(pred, y)
criterion.backward()
```

## Architecture

```
neurosketch/
├── layers/
│   └── linear.py
├── activations/
│   ├── relu.py
│   ├── sigmoid.py
│   ├── softmax.py
│   └── ...
├── losses/
│   ├── mse.py
│   ├── bce.py
│   └── sparse_cce.py
├── optimizers/
│   ├── gd.py
│   ├── sgd.py
│   └── adam.py
└── _module.py
```

## Design

Each component handles its own gradient computation:
- **Linear layer:** Computes dW, db via matrix multiplication
- **Activations:** Apply element-wise or Jacobian-based gradient chaining
- **Loss functions:** Compute gradients w.r.t. predictions
- **Optimizers:** Update weights using accumulated gradients

## Requirements

- Python 3.7+
- NumPy

## License

MIT

---

Built from scratch to understand ML fundamentals.