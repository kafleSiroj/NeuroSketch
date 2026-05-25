import numpy as np

class Optim:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def step(self):
        self._call_back()
        raise NotImplementedError

    def _call_back(self):
        for layer in self.model.layers[::-1]:
            g = layer.backward_()

class SGD(Optim):
    def __init__(self):
        super().__init__()

    def step(self):
        # return params - self.lr*grad
        pass


class MOMENTUM(Optim):
    def __init__(self, m, beta=0.9):
        super().__init__()

    def step(self):
        #     m = beta*m + (1-beta)*self.grads
        #     self.params -= self.lr*m
        pass


class ADAM(Optim):
    def __init__(self):
        super().__init__()
        self.t = 1
        
    def step(self):
        # m = beta*m + (1-beta)*self.grads
    #   v = gamma*v + (1-gamma)*(self.grads**2)

      #bias correction
    #   m_hat = m / (1-beta**self.t)
    #   v_hat = v / (1-gamma**self.t)

    #    params -= self.lr*(m_hat / (np.sqrt(v_hat) + self.epsilon))    
    #   self.t += 1
        pass
