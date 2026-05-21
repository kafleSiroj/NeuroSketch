import numpy as np

class DataLoader:
    def __init__(self, x, y, batch_size=None, shuffle=False):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if batch_size is None:
            batch_size = x.shape[0]
        
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Number of sample mismatch: {x.shape[0]} vs {y.shape[0]}")

        if batch_size <= 0 or batch_size > x.shape[0]:
            raise ValueError(f"Batch size {batch_size} must be between 1 and {x.shape[0]}")

        self.x = x.copy()
        self.y = y.copy()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_samples = self.x.shape[0]

    def _shuffle(self):
        indices = np.random.permutation(self.n_samples)
        self.x = self.x[indices]
        self.y = self.y[indices]

    def _create_batches(self):
        x_data = self.x
        y_data = self.y

        batches = []
        for i in range(0, self.n_samples, self.batch_size):
            x_batch = x_data[i: i + self.batch_size]
            y_batch = y_data[i: i + self.batch_size]
            batches.append((x_batch, y_batch))

        return batches
    
    def __call__(self):
        if self.shuffle:
            self._shuffle()
        
        self.data = self._create_batches()
        return self.data
    

class Module:
    def __init__(self):
        self.params = {}
        self.layers = []
        self.out = None
        self.input = None
        self._backward = None

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
    
    def _register_params(self, weight, bias): 
        if "weights" not in self.params:
            self.params["weights"] = []
            self.params["biases"] = []
        
        self.params["weights"].append(weight) 
        self.params["biases"].append(bias)

    def _backward(self):
        raise NotImplementedError
