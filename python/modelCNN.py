import numpy as np
from utils import conv2d, max_pool2d, relu, flatten_feature_maps, softmax, relu_derivative, cross_entropy_loss

class NetworkCNN:
    def __init__(self, conv_kernels, dense_sizes, input_shape):
        self.num_kernels = len(conv_kernels)
        self.kernels = conv_kernels
        self.dense_sizes = dense_sizes

        # Start weights and biases for dense layers
        self.weights = []
        self.biases

        # Start weights and biases for convolutional layers
        prev_size = input_shape

        for size in dense_sizes:
            w = np.random.randn(size, prev_size) * np.sqrt(2. / prev_size)
            b = np.zeros((size, 1))
            self.weights.append(w)
            self.biases.append(b)
            prev_size = size

    
        
    def forward(self, x):
        self.input = x
        self.feature_maps = [relu(conv2d(x, k)) for k in self.kernels]
        self.pooled_maps = [max_pool2d(fm) for fm in self.feature_maps]

        self.flat = flatten_feature_maps(self.pooled_maps)
        self.zs = []
        self.activations = [self.flat]

        a = self.flat
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            a = softmax(z) if W is not self.weights[-1] else relu(z)
            self.zs.append(z)
            self.activations.append(a)

        return a
    
    def backward(self, y_true, learning_rate):
        grads_w = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Backpropagation dense layers
        L = len(self.weights)
        delta = self.activations[-1] - y_true

        for l in reversed(range(L)):
            grads_w[l] = np.dot(delta, self.activations[l].T)
            grads_b[l] = delta

            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * relu_derivative(self.zs[l - 1])

        # update weights and biases
        for l in range(L):
            self.weights[l] -= learning_rate * grads_w[l]
            self.biases[l] -= learning_rate * grads_b[l]

        # Backpropagation convolutional layers 
        grad_flat = delta

        # Revert flattn to get gradients for feature maps
        pooled_grads = []
        idx = 0
        for pm in self.pooled_maps:
            size = pm.size
            grad = grad_flat[id:idx+size].reshape(pm.shape)
            pooled_grads.append(grad)
            idx += size
        
        # Revert max pooling
        relu_grads = []
        for grad, fmap in zip(pooled_grads, self.feature_maps):
            h, w = fmap.shape
            unpooled = np.zeros((h, w))
            for (i, j, (pi, pj)) in self.pool_indices:
                unpooled[i*2 + pi, j*2 + pj] = grad[i, j] # Pool size is 2x2, stride is 2
            relu_grads.append(unpooled * relu_derivative(fmap))

        # Derivative respect each kernel
        for i in range(len(self.kernels)):
            input_patch = self.input
            kernel_grad = np.zeros_like(self.kernels[i])
            kh, kw = self.kernels[i].shape
            for r in range(kernel_grad.shape[0]):
                for c in range(kernel_grad.shape[1]):
                    region = input_patch[r:r+relu_grads[i].shape[0], c:c+relu_grads[i].shape[1]]
                    if region.shape == relu_grads[i].shape:
                        kernel_grad[r, c] = np.sum(region * relu_grads[i])
            self.kernels[i] -= learning_rate * kernel_grad
        
        return self.kernels
    
    def train(self, x, y, learning_rate):
        output = self.forward(x)
        loss = cross_entropy_loss(output, y)
        self.backward(y, learning_rate)
        return loss, output


