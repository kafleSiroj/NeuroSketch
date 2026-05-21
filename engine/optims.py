import numpy as np

class Optim:
    def __init__(self, params, grads, lr=1e-2):
        self.grads = grads
        self.params = params
        self.lr = lr
        self.epsilon = 1e-4
        self.t = 1

    def SGD(self):
        self.params-=self.lr*self.grads

    def MOMENTUM(self, m, beta=0.9):
        m = beta*m + (1-beta)*self.grads
        self.params -= self.lr*m
            
    def ADAM(self, m, v, beta=0.9, gamma=0.99):
        m = beta*m + (1-beta)*self.grads
        v = gamma*v + (1-gamma)*(self.grads**2)

        #bias correction
        m_hat = m / (1-beta**self.t)
        v_hat = v / (1-gamma**self.t)

        params -= self.lr*(m_hat / (np.sqrt(v_hat) + self.epsilon))    
        self.t += 1