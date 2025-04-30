import numpy as np
from utils import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax

class Network:
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

    def forward(self, x):
        a = x
        self.zs = []
        self.activations = [a]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            self.zs.append(z)

            if i == self.num_layers - 2:
                # Última capa: softmax
                a = softmax(z)
            else:
                if self.activation_function == 'relu':
                    a = relu(z)
                else:
                    a = sigmoid(z)

            self.activations.append(a)

        return a

    def backward(self, y):
        L = self.num_layers - 1
        m = y.shape[1]

        y_hat = self.activations[-1]
        delta = y_hat - y  # Softmax + cross-entropy simplifica

        self.deltas = [None] * L
        self.deltas[-1] = np.nan_to_num(delta, nan=0.0, posinf=1e12, neginf=-1e12)

        for l in range(L - 2, -1, -1):
            z = self.zs[l]

            if self.activation_function == 'relu':
                activation_derivative = relu_derivative(z)
            else:
                activation_derivative = sigmoid_derivative(z)

            delta = np.dot(self.weights[l + 1].T, delta) * activation_derivative

            # Sanear posibles NaNs o Inf
            delta = np.nan_to_num(delta, nan=0.0, posinf=1e12, neginf=-1e12)

            if np.isnan(delta).any() or np.isinf(delta).any():
                print(f"⚠️ NaN o Inf detectado en delta de la capa {l}")
                exit()

            self.deltas[l] = delta

    def update_weights(self, learning_rate):
        L = self.num_layers - 1
        m = self.activations[0].shape[1]  # batch size

        for l in range(L):
            a_prev = self.activations[l]
            delta = self.deltas[l]

            dw = np.dot(delta, a_prev.T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m

            # Sanear posibles gradientes malos
            dw = np.nan_to_num(dw, nan=0.0, posinf=1e12, neginf=-1e12)
            db = np.nan_to_num(db, nan=0.0, posinf=1e12, neginf=-1e12)

            self.weights[l] -= learning_rate * dw
            self.biases[l] -= learning_rate * db

    def train(self, x, y, learning_rate):
        self.forward(x)
        self.backward(y)
        self.update_weights(learning_rate)
        return self.activations[-1]
