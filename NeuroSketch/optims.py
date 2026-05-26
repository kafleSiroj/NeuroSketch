import numpy as np

class Optim:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def _call_back(self):
        lyrs = self.model.layers
        last_grad = lyrs[-1].grad_next

        lin_lay = []
        for layer in self.model.layers[::-1]:
            last_grad = layer.backward_(last_grad)

            if hasattr(layer, "dW") and hasattr(layer, "dB"):
                lin_lay.append(layer)

    @staticmethod
    def _up_mov(curr_mov, grad, param, ty):
        if ty=="m":
            return param*curr_mov + (1-param)*grad

        elif ty=="v":
            return param*curr_mov + (1-param)*(grad**2)


class SGD(Optim):
    def step(self):
        self.lin_lays = self._call_back()
        for linear in self.lin_lays:
            dW, dB = linear.dW, linear.dB

            linear.params["weights"] -= self.lr*dW
            linear.params["biases"] -= self.lr*dB
        


class MOMENTUM(Optim):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.m = {"weights":[np.zeros_like(lin.params["weights"]) for lin in self.lin_lays], 
                  "biases":[np.zeros_like(lin.params["biases"]) for lin in self.lin_lays]}

    def step(self):
        self.lin_lays = self._call_back()
        for l in range(len(self.lin_lays)):
            #for weights
            self.m["weights"][l] = self._up_mov(self.m["weights"], self.lin_lays[l].dW, self.beta, "m")
            self.lin_lays[l].params["weights"] -= self.lr*self.m["weights"][l]

            #for biases
            self.m["biases"][l] = self._up_mov(self.m["biases"], self.lin_lays[l].dB, self.beta, "m")
            self.lin_lays[l].params["biases"] -= self.lr*self.m["biases"][l]


class ADAM(Optim):
    def __init__(self, beta=0.9, gamma=0.99):
        self.beta = beta
        self.gamma = gamma
        self.t = 0
        self.epsilon = 1e-4
        self.m = {"weights":[np.zeros_like(lin.params["weights"]) for lin in self.lin_lays], 
                  "biases":[np.zeros_like(lin.params["biases"]) for lin in self.lin_lays]}
 
        self.v = {"weights":[np.zeros_like(lin.params["weights"]) for lin in self.lin_lays], 
                  "biases":[np.zeros_like(lin.params["biases"]) for lin in self.lin_lays]}

    def step(self):
        self.t += 1
        self.lin_lays = self._call_back()
        for l in range(len(self.lin_lays)):
            #for weights
            self.m["weights"][l] = self._up_mov(self.m["weights"], self.lin_lays[l].dW, self.beta, "m")
            self.v["weights"][l] = self._up_mov(self.v["weights"], self.lin_lays[l].dW, self.gamma, "v")
            m_hat = self.m["weights"][l] / (1-self.beta**self.t)
            v_hat = self.v["weights"][l] / (1-self.gamma**self.t)

            self.lin_lays[l].params["weights"] -= self.lr*(m_hat / (np.sqrt(v_hat) + self.epsilon))

            #for biases            
            self.m["biases"][l] = self._up_mov(self.m["biases"], self.lin_lays[l].dB, self.beta, "m")
            self.v["biases"][l] = self._up_mov(self.v["biases"], self.lin_lays[l].dB, self.gamma, "v")
            m_hat = self.m["biases"][l] / (1-self.beta**self.t)
            v_hat = self.v["biases"][l] / (1-self.gamma**self.t)

            self.lin_lays[l].params["biases"] -= self.lr*(m_hat / (np.sqrt(v_hat) + self.epsilon))
            
            
    
