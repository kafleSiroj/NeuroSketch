import numpy as np
from NeuroSketch.engine.nn import *
from NeuroSketch.engine.act import *
from NeuroSketch.optims import *
from NeuroSketch.losses import *
from NeuroSketch.utils import DataLoader

model = Sequential(
    Linear(1,5),
    # ReLU(),
    Linear(5,10),
    Linear(10,1),        
    Sigmoid()
)

x = np.linspace(-1,1, 10)
y = np.array([1,0]*5)

load = DataLoader(x,y)
loaded = load()

optim = ADAM(model)
criterion = BinaryCrossentropyLoss(model.layers[-1])

for _ in range(10):
    for x_b, y_b in loaded:
        pred = model(x_b)
        loss = criterion(pred, y_b)
        print(loss)

        criterion.backward()
        optim.step()
    