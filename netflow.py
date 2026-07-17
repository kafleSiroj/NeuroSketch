from NeuroSketch.engine.nn import *
from NeuroSketch.engine.act import *
from NeuroSketch.losses import *
from NeuroSketch.optims import *

'''
model_dict format:

{"sequential":("<layer1>", "<layer2>", ...), "loss_fn": "<criterion>", "optimizer_alg": "<optim>"}
'''

lay_map = {"lin":  Linear,
           "relu": ReLU,
           "lrelu": LeakyReLU,
           "softmax": Softmax,
           "sigmoid": Sigmoid,
           "tanh": Tanh,
           "swish":Swish,
           "heavyside":HeavySide}

optim_map = {"sgd": SGD,
             "momentum": MOMENTUM,
             "adam": ADAM}

loss_map = {"mse": MSELoss,
            "mae": MAELoss,
            "bce": BinaryCrossentropyLoss,
            "scce": SparseCategoricalCrossentropyLoss}


class NetFlow:
    def  __init__(self, model_dict, in_out, inits):
        self.model_dict = model_dict
        self.in_out = in_out
        self.inits = inits

        self.lay_tuple = model_dict["sequential"]
        self.loss_fn = model_dict["loss_fn"]
        self.optim_alg = model_dict["optimizer_alg"]
        
        self.opt_alg_str = self.optim_alg[0]
        self.beta = self.optim_alg[1]
        self.gamma = self.optim_alg[2]

        self.model = Sequential()
        self.criterion = None
        self.optim = None
        self.losses = []


    def modelify(self):
        lin_cnt = 0
        for layer in self.lay_tuple:
            if layer == "lin":
                curr_lay = lay_map[layer](self.in_out[lin_cnt][0], self.in_out[lin_cnt][1], self.inits[lin_cnt])
                lin_cnt += 1
            else:
                curr_lay = lay_map[layer]()

            self.model.add(curr_lay)
        
        

    def trainer(self, lr, epochs, train_data):
        self.criterion = loss_map[self.loss_fn](self.model.layers[-1])
        if self.beta and self.gamma:
            self.optim = optim_map[self.optim_alg](self.model, lr, self.beta, self.gamma)
        elif self.beta or self.gamma:
            self.optim = optim_map[self.optim_alg](self.model, lr, self.beta)
        else:
            self.optim = optim_map[self.optim_alg](self.model, lr)

        for epoch in range(epochs):
            loss = []
            for x_b, y_b in train_data:
                pred = self.model(x_b)
                l = self.criterion(pred, y_b)
                loss.append(l)

                self.criterion.backward()
                self.optim.step()

            self.losses.append(sum(loss))