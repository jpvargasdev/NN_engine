import numpy as np
from utils import conv2d, max_pool2d, relu, flatten_feature_maps, softmax, relu_derivative, cross_entropy_loss, unpool2d

class NetworkCNN:
    def __init__(self, conv_kernels, dense_sizes, input_size):
        """
        conv_kernels: list of numpy arrays (e.g. 3x3 filters)
        dense_sizes: list like [64, 10]
        input_size: flattened size after pooling (e.g. 13*13*3 = 507)
        """

        # --- Convolution ---
        self.kernels = conv_kernels                    # list of 2D filters
        self.num_kernels = len(conv_kernels)

        # --- Dense layers ---
        self.dense_sizes = dense_sizes
        self.weights = []
        self.biases = []

        prev_size = input_size  # size of flattened pooled features

        for size in dense_sizes:
            # He initialization
            w = np.random.randn(size, prev_size) * np.sqrt(2. / prev_size)
            b = np.zeros((size, 1))
            self.weights.append(w)
            self.biases.append(b)
            prev_size = size

        # --- Placeholders for intermediate values (forward) ---
        self.input = None
        self.feature_maps = []
        self.pooled_maps = []
        self.pool_indices = []
        self.flat = None
        self.zs = []            # z = Wx + b pre-activations
        self.activations = []   # activations for each layer

    
        
    def forward(self, x):
        """
        x: input image of shape (H, W)
        Returns: output probabilities (softmax)
        """
        self.input = x  # Save input for backprop

        # --- Convolution → ReLU ---
        self.feature_maps = [relu(conv2d(x, kernel)) for kernel in self.kernels]

        # --- Max Pooling ---
        self.pooled_maps = []
        self.pool_indices = []
        for fmap in self.feature_maps:
            pooled, indices = max_pool2d(fmap)
            self.pooled_maps.append(pooled)
            self.pool_indices.append(indices)

        # --- Flatten pooled maps ---
        self.flat = flatten_feature_maps(self.pooled_maps)  # shape: (N, 1)

        # --- Dense layers ---
        self.zs = []
        self.activations = [self.flat]

        a = self.flat
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(W, a) + b
            a = relu(z) if idx < len(self.weights) - 1 else softmax(z)
            self.zs.append(z)
            self.activations.append(a)

        return a  # softmax output
    
    def backward(self, y_true, learning_rate):
        """
        Backpropagation through dense layers and convolutional kernels.
        y_true: (10, 1) — one-hot label
        """

        # --- Paso 1: error en capa de salida (softmax + cross-entropy) ---
        L = len(self.weights)
        y_hat = self.activations[-1]
        delta = y_hat - y_true  # (10, 1)

        # --- Paso 2: calcular gradientes de capas densas ---
        grads_w = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        for l in reversed(range(L)):
            a_prev = self.activations[l]  # activación anterior
            grads_w[l] = np.dot(delta, a_prev.T)
            grads_b[l] = delta

            if l > 0:
                z_prev = self.zs[l - 1]
                delta = np.dot(self.weights[l].T, delta) * relu_derivative(z_prev)
                delta = np.nan_to_num(delta, nan=0.0, posinf=1e12, neginf=-1e12)

        # --- Paso 3: retropropagar hacia los feature maps (desde capa densa a flatten) ---
        delta = np.dot(self.weights[0].T, delta)  # (507, 1)
        grad_flat = delta

        # --- Paso 4: aplicar actualizaciones en capas densas ---
        for l in range(L):
            self.weights[l] -= learning_rate * grads_w[l]
            self.biases[l]  -= learning_rate * grads_b[l]

        # --- Paso 5: reconstruir los pooled_grads desde grad_flat ---
        pooled_grads = []
        idx = 0
        for pm in self.pooled_maps:
            size = pm.size
            grad_slice = grad_flat[idx:idx + size]
            grad = grad_slice.reshape(pm.shape)
            pooled_grads.append(grad)
            idx += size

        # --- Paso 6: unpooling y ReLU backward ---
        relu_grads = []
        for grad, fmap, indices in zip(pooled_grads, self.feature_maps, self.pool_indices):
            unpooled = unpool2d(grad, indices, fmap.shape)
            relu_back = unpooled * relu_derivative(fmap)
            relu_grads.append(relu_back)

        # --- Paso 7: calcular y aplicar gradientes a cada kernel ---
        for i in range(len(self.kernels)):
            kernel_grad = np.zeros_like(self.kernels[i])
            grad_map = relu_grads[i]
            input_patch = self.input  # imagen original
            kh, kw = self.kernels[i].shape

            for r in range(kernel_grad.shape[0]):
                for c in range(kernel_grad.shape[1]):
                    region = input_patch[r:r+grad_map.shape[0], c:c+grad_map.shape[1]]
                    if region.shape == grad_map.shape:
                        kernel_grad[r, c] = np.sum(region * grad_map)

            self.kernels[i] -= learning_rate * kernel_grad
        
    def train(self, x, y, learning_rate):
        output = self.forward(x)
        loss = cross_entropy_loss(output, y)
        self.backward(y, learning_rate)
        return loss, output


