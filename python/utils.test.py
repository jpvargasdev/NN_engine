import numpy as np
from utils import conv2d, max_pool2d, relu

input_img = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])
print("Input")
print(input_img)

print("Kernel")
print(kernel)
output = conv2d(input_img, kernel)
print("Convolutional Layer")
print(output)

print("Nonlinearity")
relu_output = relu(output)
print(relu_output)

maxpool = max_pool2d(relu_output)
print("Max Pooling")
print(maxpool)
