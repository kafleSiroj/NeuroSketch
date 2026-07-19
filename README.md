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
- SGD, MOMENTUM, ADAM

## Installation

```bash
pip install NeuroSketch
```


## Quick Start

### Binary Classification

```python
import numpy as np
from NeuroSketch.engine.nn import Sequential, Linear
from NeuroSketch.engine.act import Sigmoid
from NeuroSketch.losses import BinaryCrossentropyLoss
from NeuroSketch.optims import Adam
from NeuroSketch.utils import DataLoader

# Create data
x = np.random.randn(100, 10)
y = np.random.randint(2, size=100)
loader = DataLoader(x, y, batch_size=16, shuffle=True)

# Create model
model = Sequential(
    Linear(in_features=10, out_features=1, init_type="he"),
    Sigmoid()
)
criterion = BinaryCrossentropyLoss(model.layers[-1])
optimizer = Adam(model, lr=0.01)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in loader:
        # Forward
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        
        # Backward
        criterion.backward()
        
        # Update
        optimizer.step()
```

### Multiclass Classification

```python
from NeuroSketch.engine.nn import Sequential, Linear
from NeuroSketch.engine.act import Softmax
from NeuroSketch.losses import SparseCategoricalCrossentropyLoss
from NeuroSketch.optims import Adam
from NeuroSketch.utils import DataLoader

# Create model
model = Sequential(
    Linear(in_features=10, out_features=3, init_type="he"),
    Softmax()
)
criterion = SparseCategoricalCrossentropyLoss(model.layers[-1])
optimizer = Adam(model, lr=0.01)

# Train
loader = DataLoader(x, y, batch_size=16, shuffle=True)
for epoch in range(10):
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        criterion.backward()
        optimizer.step()
```

## Architecture

```
NeuroSketch/
├── engine/
│   ├── nn.py         # Linear, Sequential
│   └── act.py        # All activations
├── losses.py         # Loss functions
├── optims.py         # Optimizers (SGD, Momentum, Adam)
└── utils.py          # DataLoader
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