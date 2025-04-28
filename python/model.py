import numpy as np
from utils import sigmoid, sigmoid_derivative, relu, relu_derivative

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)

        self.weights = []
        self.biases = []

        # Random init weights and biasses
        for i in range(self.num_layers - 1):
                        
            w = np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2. / layers[i])

            w = np.random.randn(layers[i+1], layers[i]) / np.sqrt(layers[i])
            b = np.zeros((layers[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        a = x
        self.zs = []
        self.activations = [a]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)

        return a

    def backward(self, y):
        L = self.num_layers - 1
