import numpy as np

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    # we use clip because if x is a very large number e.g x = -1000 then np.exp(x) will crash
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
    
def relu(x):
    return np.maximum(0, x)

class Network: 
    def __init__(self, layers, biases, weights):
        self.layers = layers
        self.weights = weights
        self.biases = biases
        self.num_layers = len(layers)

    def forward(self, entries):
        activation = entries
        self.preactivations = []
        self.activations = [activation]

        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            ## Basic preactivation formula z = w * x + b
            preactivation = np.dot(weight, activation) + bias            
            self.preactivations.append(preactivation)

            ## Basic activation f(z) in this case we use sigmoid
            activation = sigmoid(preactivation)
            self.activations.append(activation)

        return activation

    def backward(self, desired_outputs):
        ## Last layer
        L = self.num_layers - 1
        ## Prediction / last activation
        y_hat = self.activations[-1]
        ## Last layer error
        delta_output = (y_hat - desired_outputs) * sigmoid_derivative(self.preactivations[-1])
        self.deltas = [None] * self.num_layers
        self.deltas[L] = delta_output

        ## Backpropagation hidden layers
        for layer_index in range(L -1, 0, -1):
            W_next_T = self.weights[layer_index].T
            delta_next = self.deltas[layer_index + 1]
            z = self.preactivations[layer_index - 1]
            activation_derivative = sigmoid_derivative(z)
            delta = np.dot(W_next_T, delta_next) * activation_derivative

            # Sanitize gradients to avoid NaN or infinite values (e.g. from numerical instability)
            delta = np.nan_to_num(delta, nan=0.0, posinf=1e12, neginf=-1e12)
            
            self.deltas[layer_index] = delta

    def update_weights(self, learning_rate):
        L = self.num_layers - 1  # Number of layers with weights (excluding input)
        m = self.activations[0].shape[0]  # Batch size = number of input samples
    
        for l in range(L):
            a_prev = self.activations[l].reshape(-1, 1)  # activation from previous layer
            delta = self.deltas[l + 1].reshape(-1, 1)     # delta of current layer
    
            # Compute gradient of the weights: dW = delta * a_prev.T / m
            dw = np.dot(delta, a_prev.T) / m
    
            # Compute gradient of the biases: db = sum of deltas / m
            db = np.sum(delta, axis=1, keepdims=True) / m
    
            # Sanitize gradients to avoid NaN or infinite values (e.g. from numerical instability)
            dw = np.nan_to_num(dw, nan=0.0, posinf=1e12, neginf=-1e12)
            db = np.nan_to_num(db, nan=0.0, posinf=1e12, neginf=-1e12)
    
            # Update weights and biases using gradient descent
            self.weights[l] -= learning_rate * dw
            self.biases[l] -= learning_rate * db


    def train(self, entries, desired_outputs, learning_rate):
        self.forward(entries=entries)
        self.backward(desired_outputs=desired_outputs)
        self.update_weights(learning_rate)
        return self.activations[-1]
    

# XOR Dataset
X = [np.array([[0], [0]]),
     np.array([[0], [1]]),
     np.array([[1], [0]]),
     np.array([[1], [1]])]

Y = [np.array([[0]]),
     np.array([[1]]),
     np.array([[1]]),
     np.array([[0]])]

# Single-layer Network: [2 input - 1 output]
single_layers = [2, 1]
single_weights = [np.random.randn(1, 2)]
single_biases = [np.random.randn(1, 1)]

# Multi-layer Network: [2 input → 2 hidden → 1 output]
layers = [2, 2, 1]
weights = [np.random.randn(2, 2), np.random.randn(1, 2)]
biases = [np.random.randn(2, 1), np.random.randn(1, 1)]

# Create instance using your Network class
single_net = Network(layers=single_layers, weights=single_weights, biases=single_biases)
net = Network(layers=layers, weights=weights, biases=biases)

# Training loop
epochs = 10000
lr = 0.1

for epoch in range(epochs):
    for x, y in zip(X, Y):
        single_net.train(x, y, learning_rate=lr)
        net.train(x, y, learning_rate=lr)

# Evaluation
print("Final predictions after training on XOR single-layer:")
for x, y in zip(X, Y):
    single_output = single_net.forward(x)
    print(f"Single Layer: Input: {x.T}, Expected: {y.item()}, Output: {single_output[0][0]:.3f}")

print("Final predictions after training on XOR multi-layer:")
for x, y in zip(X, Y):
    output = net.forward(x)
    print(f"Multi-Layer: Input: {x.T}, Expected: {y.item()}, Output: {output[0][0]:.3f}")