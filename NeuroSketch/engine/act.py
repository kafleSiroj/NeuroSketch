import numpy as np
from _module import Module

class Activation(Module):
    def forward(self, x):
        raise NotImplementedError
    
    def _backward(self):
        raise NotImplementedError
    
    
class ReLU(Activation):
    def forward(self, x):
        val = np.maximum(0, x)
        self.input = x
        self.out = val
        return val
    
    def _backward(self):
        grad = np.where(self.input > 0.0, 1.0, 0.0)
        return grad          
    
    def __repr__(self):
        return "ReLU()"
    

class Sigmoid(Activation):
    def forward(self, x):
        val = 1 / (1 + np.exp(-x))
        self.input = x
        self.out = val
        return val
    
    def _backward(self):
        grad = self.out*(1-self.out)
        return grad
    
    def __repr__(self):
        return "Sigmoid()"
    

class Softmax(Activation):
    def forward(self, x):
        final_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(final_x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.input = x
        self.out = probs
        return probs
    
    def _backward(self):
        jac = []
        for prob in self.out:
            grad = np.diag(prob) - np.outer(prob, prob)
            jac.append(grad)

        return np.array(jac)
    
    def __repr__(self):
        return "Softmax()"
    

class Tanh(Activation):
    def forward(self, x):
        val = np.tanh(x)
        self.input = x
        self.out = val
        return val
    
    def _backward(self):
        grad = 1 - self.out**2
        return grad
    
    def __repr__(self):
        return "Tanh()"


class HeavySide(Activation):
    def forward(self, x):
        val = np.where(x > 0.0, 1.0, 0.0)
        self.input = x
        self.out = x
        return val
    
    def _backward(self):
        return np.zeros_like(self.input)

    def __repr__(self):
        return "HeavySide()"


class Swish(Activation):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        val = x * (1 / (1 + np.exp(-x*self.beta)))
        self.input = x
        self.out = x
        return val
    
    def _backward(self):
        sig_val = 1 / (1 + np.exp(-self.beta*self.input))
        sig_grad = sig_val*(1-sig_val)

        return sig_val + self.beta * self.input * sig_grad

    def __repr__(self):
        return f"Swish(beta={self.beta})"
    

class LeakyReLU(Activation):
    def __init__(self, alpha=1e-4):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        val = np.where(x > 0.0, x, self.alpha*x)
        self.input = x
        self.out = x
        return val
    
    def _backward(self):
        grad = np.where(self.input > 0.0, 1.0, self.alpha)
        return grad

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"
    
