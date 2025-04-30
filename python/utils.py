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
