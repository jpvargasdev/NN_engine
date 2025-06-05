import numpy as np
import matplotlib.pyplot as plt

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
    """
    Computes the softmax function for the given input array.

    The softmax function is often used in the final layer of a neural network
    when the task is a multi-class classification problem.

    Parameters:
        z (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The softmax output of the input array.
    """
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

def one_hot_encode_cnn(y, num_classes=10):
    if y.ndim == 2:
        y = y.flatten()
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

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

def max_pool2d(feature_map, size=2, stride=2):
    h, w = feature_map.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    pooled = np.zeros((out_h, out_w))
    indices = {}  # guardaremos (i, j) → (pi, pj)

    for i in range(out_h):
        for j in range(out_w):
            patch = feature_map[i*stride:i*stride+size, j*stride:j*stride+size]
            max_val = np.max(patch)
            max_pos = np.unravel_index(np.argmax(patch), patch.shape)  # (pi, pj)
            pooled[i, j] = max_val
            indices[(i, j)] = max_pos  # ✅ esto es clave

    return pooled, indices

def flatten_feature_maps(feature_maps):
    return np.vstack([fm.flatten().reshape(-1, 1) for fm in feature_maps])

def show_predictions(X, Y_true, Y_pred, n=20):
    plt.figure(figsize=(15, 4))
    for i in range(n):
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(X[i], cmap='gray')
        plt.axis('off')
        color = 'green' if Y_true[i] == Y_pred[i] else 'red'
        plt.title(f"P:{Y_pred[i]} / T:{Y_true[i]}", color=color)
    plt.tight_layout()
    plt.show()

def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch (logged)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def plot_accuracy_over_time(acc_history):
    plt.plot(acc_history)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def unpool2d(pooled_grad, indices, original_shape, size=2, stride=2):
    """
    pooled_grad: gradiente de salida del max_pool (shape: pooled feature map)
    indices: dict de (i, j) → (pi, pj) con la posición del máximo original
    original_shape: shape del feature map original antes del pooling
    """
    h, w = original_shape
    unpooled = np.zeros((h, w))

    for (i, j), (pi, pj) in indices.items():
        row = i * stride + pi
        col = j * stride + pj
        unpooled[row, col] = pooled_grad[i, j]

    return unpooled

def show_kernels(kernels):
    """
    Muestra una visualización de los kernels convolucionales (filtros).
    """
    num_kernels = len(kernels)
    plt.figure(figsize=(2 * num_kernels, 2))
    for i, kernel in enumerate(kernels):
        plt.subplot(1, num_kernels, i + 1)
        plt.imshow(kernel, cmap='gray')
        plt.title(f"Kernel {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    