
# NeuroSketch

> A lightweight educational neural network framework built from scratch with NumPy.

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![NumPy](https://img.shields.io/badge/NumPy-Required-orange)
![License](https://img.shields.io/badge/License-MIT-green)

NeuroSketch is an educational deep learning framework that implements neural networks from first principles. Every stage of training—from forward propagation to gradient computation and parameter updates—is written manually using NumPy.

Unlike production frameworks that rely on automatic differentiation, NeuroSketch makes every step explicit. Layers compute their own gradients, loss functions generate the initial gradient, and optimizers traverse the network during backpropagation. The goal is to understand *how* neural networks learn rather than simply using them.

---

# Features

## Layers

- Fully Connected (`Linear`)
- Sequential model container

## Activation Functions

- ReLU
- LeakyReLU
- Sigmoid
- Tanh
- Softmax
- Swish
- HeavySide

## Loss Functions

- MSELoss
- MAELoss
- BinaryCrossentropyLoss
- SparseCategoricalCrossentropyLoss

## Optimizers

- SGD
- MOMENTUM
- ADAM

## Utilities

- Mini-batch `DataLoader`
- Dataset shuffling
- Optional dropping of incomplete batches

## Weight Initialization

- He
- Xavier
- Zero

---

# Installation

```bash
pip install NeuroSketch
```

---

# Quick Example

```python
import numpy as np

from NeuroSketch.engine.nn import Sequential, Linear
from NeuroSketch.engine.act import ReLU, Softmax
from NeuroSketch.losses import SparseCategoricalCrossentropyLoss
from NeuroSketch.optims import ADAM
from NeuroSketch.utils import DataLoader

x = np.random.randn(500, 20)
y = np.random.randint(5, size=500)

loader = DataLoader(x, y, batch_size=32, shuffle=True)

model = Sequential(
    Linear(20, 64, init_type="he"),
    ReLU(),
    Linear(64, 5, init_type="xavier"),
    Softmax()
)

criterion = SparseCategoricalCrossentropyLoss(model.layers[-1])
optimizer = ADAM(model, lr=1e-3)

for epoch in range(20):
    total_loss = 0

    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        criterion.backward()
        optimizer.step()

        total_loss += loss

    print(f"Epoch {epoch+1}: {total_loss:.4f}")
```

## Model Summary

```python
print(model.summary())
```

---

# Project Structure

```text
src/
└── NeuroSketch/
    ├── engine/
    │   ├── _module.py
    │   ├── act.py
    │   └── nn.py
    ├── losses.py
    ├── optims.py
    ├── utils.py
    └── LICENSE

README.md
```

---

# Framework Design

```
Forward Pass

Input
 │
Linear
 │
Activation
 │
Linear
 │
Activation
 │
Prediction
 │
Loss

Backward Pass

Loss
 │
criterion.backward()
 │
optimizer.step()
 │
Activation.backward()
 │
Linear.backward()
 │
Activation.backward()
 │
Linear.backward()
 │
Parameter Update
```

NeuroSketch follows a modular object-oriented design.

- **Layers** perform forward and backward propagation.
- **Activations** compute their own derivatives.
- **Losses** compute the initial gradient.
- **Optimizers** drive the complete backpropagation process and update parameters.

No automatic differentiation or computational graph is used.

---

# Training Flow

```python
prediction = model(x_batch)
loss = criterion(prediction, y_batch)

criterion.backward()
optimizer.step()
```

Internally:

1. Forward propagation through every layer.
2. Loss computation.
3. Initial gradient generation.
4. Optimizer walks backward through the model by calling each layer's `backward()`.
5. Parameters are updated.

---

# Components

## Linear Layer

Caches:

- Input
- Weights
- Biases

Computes:

- `dW`
- `dB`
- Gradient for the previous layer

using vectorized NumPy operations.

## Activation Functions

Each activation caches its forward output and computes its derivative during backpropagation.

Softmax implements the Jacobian-vector product for efficient gradient propagation.

## Loss Functions

Every loss object stores predictions and labels during the forward pass.

Calling

```python
criterion.backward()
```

computes the gradient of the loss with respect to the model output and passes it to the final activation layer.

## Optimizers

Optimizers not only update parameters but also perform the complete backward traversal of the network.

Implemented:

- SGD
- MOMENTUM
- ADAM

## DataLoader

Supports:

- Mini-batching
- Full-batch training
- Dataset shuffling
- Dropping incomplete batches

---

# Design Philosophy

NeuroSketch intentionally avoids hidden abstractions.

Instead of relying on automatic differentiation, every layer implements its own forward and backward computations. This exposes every mathematical operation involved in neural network training, making the framework suitable for education and experimentation.

---

# Comparison

| Feature | NeuroSketch | PyTorch |
|----------|-------------|----------|
| Built with NumPy | ✅ | ❌ |
| Manual gradient computation | ✅ | ❌ |
| Automatic differentiation | ❌ | ✅ |
| Explicit backpropagation | ✅ | ❌ |
| Educational focus | ✅ | ⚠️ |
| Production ready | ❌ | ✅ |

---

# Roadmap

## Completed

- ✅ Sequential models
- ✅ Linear layers
- ✅ Weight initialization
- ✅ Multiple activations
- ✅ Multiple loss functions
- ✅ SGD
- ✅ MOMENTUM
- ✅ ADAM
- ✅ DataLoader

## Planned

- [ ] Dropout
- [ ] Batch Normalization
- [ ] Learning-rate schedulers
- [ ] CNN layers
- [ ] Pooling layers
- [ ] RNN / LSTM
- [ ] Model save/load
- [ ] Documentation website

---

# Requirements

- Python 3.7+
- NumPy

---

# License

MIT License.

---

# Why NeuroSketch?

NeuroSketch was created with one goal:

> **Understand neural networks by building them—not by treating them as black boxes.**

Every gradient, parameter update, and layer operation is implemented manually using NumPy, making the framework a practical resource for students, educators, and anyone interested in learning deep learning from first principles.
