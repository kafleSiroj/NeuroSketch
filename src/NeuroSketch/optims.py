import numpy as np
from .engine.nn import Sequential

class Optim:
    """
        ```
        model: pass the entire model
        lr=1e-3: learning rate
        ```

        ### Usage:
        ```
        optim1 = SGD(model, lr=0.01) #batches created while loading data, works as both full batch and mini batch gradient descent

        optim2 = MOMENTUM(model, beta=0.9) #beta=0.9: Momentum parameter

        optim3 = ADAM(model, lr=0.001, beta=0.9, gamma=0.999)  #beta: first momentum; gamma: second momentum
        ```
    """
    def __init__(self, model: Sequential, lr: float):
        self.model = model
        self.lr = lr

    def step(self):
        """
        Does backward pass to every layer of model from last layer, from what it caches the chained gradient upto previous layer as `grad_next` in the current layer by calling each layer's `.backward(next_grad)`
        <br>
        <br>
        Then filters updatable layer `Linear` which has parameters `dW` and `dB`, fetches those and updates the parameters of all linear layers. This process is done my a static method `_call_back`

        
        """
        raise NotImplementedError

    def _call_back(self):
        lyrs = self.model.layers
        last_grad = lyrs[-1].grad_next

        lin_lay = []
        for layer in self.model.layers[::-1]:
            last_grad = layer.backward(last_grad)

            if hasattr(layer, "dW") and hasattr(layer, "dB"):
                lin_lay.append(layer)
        
        return lin_lay

    @staticmethod
    def _up_mov(curr_mov, grad, param, ty):
        if ty=="m":
            return param*curr_mov + (1-param)*grad

        elif ty=="v":
            return param*curr_mov + (1-param)*(grad**2)


class SGD(Optim):
    """
        ```
        model: pass the entire model
        lr=1e-3: learning rate
        ```
        *works as both batch and mini-batch*

        
        ### Usage:
        ```
            optim = SGD(model, lr=0.01) 

            for x_batch, y_batch in data_loader:
                ...
                optim.step()
                ...
        ``` 
    """

    def __init__(self, model, lr=1e-3):
        super().__init__(model, lr)

    def step(self):
        self.lin_lays = self._call_back()
        for linear in self.lin_lays:
            dW, dB = linear.dW, linear.dB

            linear.params["weights"][0] -= self.lr*dW
            linear.params["biases"][0] -= self.lr*dB
        


class MOMENTUM(Optim):
    """
        ```
        model: pass the entire model
        lr=1e-3: learning rate
        ```

        ### Usage:
        ```
            optim = MOMENTUM(model, lr=1e-3, beta=0.9) #beta=0.9: Momentum parameter

            for x_batch, y_batch in data_loader:
                ...
                optim.step()
                ...
        ```
    """
    def __init__(self, model, lr=1e-3, beta=0.9):
        super().__init__(model, lr)
        self.beta = beta
        self.m = None

    def step(self):
        self.lin_lays = self._call_back()
        if self.m is None:
            self.m = {"weights":[np.zeros_like(lin.params["weights"][0]) for lin in self.lin_lays], 
                    "biases":[np.zeros_like(lin.params["biases"][0]) for lin in self.lin_lays]}
            
        for l in range(len(self.lin_lays)):
            #for weights
            self.m["weights"][l] = self._up_mov(self.m["weights"][l], self.lin_lays[l].dW, self.beta, "m")
            self.lin_lays[l].params["weights"][0] -= self.lr*self.m["weights"][l]

            #for biases
            self.m["biases"][l] = self._up_mov(self.m["biases"][l], self.lin_lays[l].dB, self.beta, "m")
            self.lin_lays[l].params["biases"][0] -= self.lr*self.m["biases"][l]


class ADAM(Optim):
    """
        ```
        model: pass the entire model
        lr=1e-3: learning rate
        ```

        Usage:
        ```
            optim = ADAM(model, lr=0.001, beta=0.9, gamma=0.999)  #beta=0.9: first momentum; gamma=0.999: second momentum

            for x_batch, y_batch in data_loader:
                ...
                optim.step()
                ...
        ```
    """
    def __init__(self, model, lr=1e-3, beta=0.9, gamma=0.999):
        super().__init__(model, lr)
        self.beta = beta
        self.gamma = gamma
        self.t = 0
        self.epsilon = 1e-8
        self.m = None
        self.v = None

    def step(self):
        self.t += 1
        self.lin_lays = self._call_back()
        if self.m is None and self.v is None:
            self.m = {"weights":[np.zeros_like(lin.params["weights"][0]) for lin in self.lin_lays], 
                    "biases":[np.zeros_like(lin.params["biases"][0]) for lin in self.lin_lays]}
    
            self.v = {"weights":[np.zeros_like(lin.params["weights"][0]) for lin in self.lin_lays], 
                    "biases":[np.zeros_like(lin.params["biases"][0]) for lin in self.lin_lays]}
            
        for l in range(len(self.lin_lays)):
            #for weights
            self.m["weights"][l] = self._up_mov(self.m["weights"][l], self.lin_lays[l].dW, self.beta, "m")
            self.v["weights"][l] = self._up_mov(self.v["weights"][l], self.lin_lays[l].dW, self.gamma, "v")
            m_hat = self.m["weights"][l] / (1-self.beta**self.t)
            v_hat = self.v["weights"][l] / (1-self.gamma**self.t)

            self.lin_lays[l].params["weights"][0] -= self.lr*(m_hat / (np.sqrt(v_hat) + self.epsilon))

            #for biases            
            self.m["biases"][l] = self._up_mov(self.m["biases"][l], self.lin_lays[l].dB, self.beta, "m")
            self.v["biases"][l] = self._up_mov(self.v["biases"][l], self.lin_lays[l].dB, self.gamma, "v")
            m_hat = self.m["biases"][l] / (1-self.beta**self.t)
            v_hat = self.v["biases"][l] / (1-self.gamma**self.t)

            self.lin_lays[l].params["biases"][0] -= self.lr*(m_hat / (np.sqrt(v_hat) + self.epsilon))
            
            
    
