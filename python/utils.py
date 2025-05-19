import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(z):
    z = np.nan_to_num(z)  # 🔥 Reemplaza nan e inf por 0
    z_shift = z - np.max(z, axis=0, keepdims=True)  # Evitar overflow
    exp_z = np.exp(z_shift)
    sum_exp = np.sum(exp_z, axis=0, keepdims=True)

    # 🔥 Proteger contra división por cero
    sum_exp = np.clip(sum_exp, 1e-12, np.inf)

    return exp_z / sum_exp

def cross_entropy_loss(y_hat, y_true):
    m = y_true.shape[1]
    epsilon = 1e-12 
    y_hat_clipped = np.clip(y_hat, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_hat_clipped)) / m
    return loss

def one_hot_encode(y, num_classes=10):
    m = y.shape[1]
    one_hot = np.zeros((num_classes, m))
    for i in range(m):
        label = y[0, i]
        one_hot[label, i] = 1
    return one_hot

def conv2d(input, kernel, stride=1, padding=0):
    """
        2D Convolution (single channel) - no libraries

        Args:
            input: Input matrix
            kernel: Kernel matrix
            stride: Stride of the convolution
            padding: Padding of the convolution

        Returns:
            Output matrix after convolution 2D
    """

    # Padding
    if padding > 0:
        input = np.pad(input, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # Convolution
    h, w = input.shape
    kh, kw = kernel.shape

    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1

    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            # region = input[i*stride : i*stride+kh, j*stride : j*stride+kw]
            region = input[i*stride : i*stride+kh, j*stride : j*stride+kw]
            output[i, j] = np.sum(region * kernel)

    return output

def max_pool2d(input, kernel_size=2, stride=2):
    """
    Apply Max pooling 2d on an 2d array (one layer)

    Args:
        input: Input matrix
        kernel_size: Size of the kernel
        stride: Stride of the pooling

    Returns:
        Output matrix after pooling
    """
    h, w = input.shape
    oh = (h - kernel_size) // stride + 1
    ow = (w - kernel_size) // stride + 1

    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            region = input[i*stride : i*stride+kernel_size, j*stride : j*stride+kernel_size]
            output[i, j] = np.max(region)

    return output
