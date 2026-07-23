
# NeuroSketch


![Python](https://img.shields.io/badge/Python-3.7+-blue)
![NumPy](https://img.shields.io/badge/NumPy-Required-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![PyPI version](https://img.shields.io/pypi/v/NeuroSketch)


NeuroSketch is a lightweight educational deep learning framework that implements neural networks from first principles. Every stage of training, from forward propagation to backpropagation, is written manually using NumPy.

Here is a demo, comparing with PyTorch:

<a target="_blank" href="https://colab.research.google.com/github/kafleSiroj/NeuroSketch/blob/main/NeuroSketch%20vs%20PyTorch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


---

# Features

## Layers

- Fully Connected (`Linear`)
- Sequential model container (`Sequential`)
- Activation Functions
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
- `DataLoader`
    - Batch createion
    - Dataset shuffling
    - Optional dropping of incomplete batches

## Weight Initialization

- He
- Xavier
- Zero
- Random (Default)

---

# Installation

```bash
pip install neurosketch
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
    Linear(20, 64),
    ReLU(),
    Linear(64, 5),
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
│   ├── engine/
│   │   ├── _module.py
│   │   ├── act.py
│   │   └── nn.py
│   ├── losses.py
│   ├── optims.py
│   ├── utils.py
│   └── LICENSE
└──README.md
```

---

# Framework Design

```
Forward Pass

Input
 │
Model
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
*layers.backward()
 |
Parameter Update
```

NeuroSketch follows a modular object-oriented design.

- **Layers** perform forward propagation and gradient computation.
- **Losses** compute the initial gradient i.e. of gradient of loss function with respect to the output of the last layer.
- **Optimizers** drive the complete backpropagation process, and update parameters.

No automatic differentiation or computational graph is used.

---

# Training Flow
The syntax in training process is same for all cases

```python
prediction = model(x_batch)
loss = criterion(prediction, y_batch)

criterion.backward()
optimizer.step()
```
---

# Model setup:

1. First, import `DataLoader` object, it is located as `neurosketch.utils`. Then, load your data as `loaded_data = DataLoader(x: numpy.ndarray, y: numpy.ndarray, batch_size=<int>, shuffle=<bool>, drop_last=<bool>)` here, 
    ``` 
    batch_size;
    if None: full-batch, if value <int> given, mini-batch

    shuffle;
    if True:
        shuffles batches every epoch
    else:
        doesn't shuffle batches

    drop_last;
    if True:
        drops incomplete batch
    else:
        doesn't drop incomplete batch        
    ```
2. Import the `Sequential` layer contatiner from `neurosketch.engine.nn`
3. Now, import the `Linear` layer object from `neurosketch.engine.nn` and essential activations from `neurosketch.engine.nn.act`
4. You can define you model in two ways:
    ```python
    model = Sequential(
        <*layers>
    )
    ```
    or
    ```python
    model = Sequential()
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    ...
    ...
    ```
    Note: The `Linear` layer requires you to enter two parameter `in_features` and `out_features` because shape of linear layer is *(in_features x out_features)*. You can even specify weight initialization for each linear layer. Shape of weight: *(out_features x in_feature)*.

5. Now, import required loss from `neurosketch.losses` and required optimizer from `neurosketch.optims`. 
6. To setup optimizer, you should pass entire model `model` into it, and you can put enter the learning rate `lr`:
    `optim = <Optimizer>(model: Sequential, lr=1e-3)` or,
for optimizers like Adam and Momentum, you can also edit the momentum parameter `beta=0.9` for `MOMENTUM`; `beta=0.9` and `gamma=0.999` for `ADAM`
7. To setup the criterion function, you should pass the last layer of model `model.layers[-1]` to define the criterion;
    `criterion = <Loss>(model.layers[-1])`
and to find the loss, you can do:
    `loss = criterion(pred, label)`
8. Now, you can iterate through epochs and `loaded_data` like:
    ```python
    for epoch in range(epochs):
        for x_batch, y_batch in loaded_data:
            ...
            ...
    ```
    and use the same trainig flow syntax mentioned above

---

# Working Principle:

1. `model(x_batch)` does forward propagation for each layers, caches and intermediate values inside each layer object
2. `criterion(prediction, y_batch)` returns the loss value
3. `critetion.backward()` calls the `.backward()` of the loss function, this calculates gradient of loss with respect to output of model `prediction`, this gets cached as `grad_next` of the last layer of the model, which is passed into the loss object during criterion defination `criterion = <Loss>(model.layers[-1])`
4. Optimizer does backward pass to every layer of model from last layer, from what it caches the chained gradient upto previous layer as `grad_next` in the current layer by calling each layer's `.backward(next_grad)`
5. Then the optimizer filters updatable layer `Linear` which has parameters `dW` and `dB`, fetches those and updates the parameters of all linear layers.

---


# Requirements

- Python 3.7+
- NumPy

---

# License

MIT License.

---

