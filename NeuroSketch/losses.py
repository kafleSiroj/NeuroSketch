import numpy as np

class Loss:
    def __init__(self):
        self.epsilon = 1e-4
        self._label = None
        self._pred = None
        self._n = None
    
    def __call__(self, pred, label):
        self._n = len(label)
        self._label = label
        return self.forward(pred, label)
    
    def forward(self, pred, label):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
    


class MSELoss(Loss):
    def forward(self, pred, label):
        loss = np.mean(np.square(pred - label))
        self._pred = pred
        return loss
    
    def backward(self):
        grad = 2 * (self._pred - self._label) / self._n
        return grad
    
    def __repr__(self):
        return "MSELoss()"
    

class MAELoss(Loss):
    def forward(self, pred, label):
        loss = np.mean(abs(label - pred))
        self._pred = pred
        return loss
    
    def backward(self):
        grad = np.sign(self._pred - self._label) / self._n
        return grad

    def __repr__(self):
        return "MAELoss()"


class BinaryCrossentropyLoss(Loss):
    def forward(self, pred, label):
        y_hat = np.clip(pred, self.epsilon, 1 - self.epsilon)
        loss = -np.mean(label*np.log(y_hat) + (1-label)*np.log(1-y_hat))
        self._pred = y_hat
        return loss
    
    def backward(self):
        grad = ((self._pred-self._label) / (self._pred*(1-self._pred))) / self._n
        return grad

    def __repr__(self):
        return f"BinaryCrossentropyLoss(epsilon={self.epsilon})"


class SparseCategoricalCrossentropyLoss(Loss):
    def forward(self, pred, label):
        y_hat  = np.clip(pred, self.epsilon, 1 - self.epsilon)
        tclss = y_hat[np.arange(self._n), label]
        loss = -np.mean(np.log(tclss))
        self._pred = (y_hat, tclss)
        return loss

    def backward(self):
        grad = np.zeros_like(self._pred[0])
        grad[np.arange(self._n), self._label] = -1 / (self._pred[1] * self._n)
        return grad

    def __repr__(self):
        return f"SparseCategoricalCrossentropyLoss(epsilon={self.epsilon})"
    
