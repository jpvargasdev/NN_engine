import numpy as np

class NetworkCNN:
    def __init__(self, layers, activation_function='sigmoid'):
        self.layers = layers
        self.num_layers = len(layers)
        self.activation_function = activation_function

        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2. / layers[i])
            b = np.zeros((layers[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
