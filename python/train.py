import numpy as np
import matplotlib.pyplot as plt
from utils import cross_entropy_loss, one_hot_encode
from model import Network

train_data = np.loadtxt('./dataset/mnist_test.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('./dataset/mnist_test.csv', delimiter=',', skiprows=1)

X = train_data[:, 1:] / 255.0              
Y = train_data[:, 0].astype(int)           
X_test = test_data[:, 1:] / 255.0
Y_test = test_data[:, 0].astype(int)


X = X.T                              
Y = Y.reshape(1, -1)             
Y = one_hot_encode(Y)            

X_test = X_test.T
Y_test = Y_test.reshape(1, -1)
Y_test_encoded = one_hot_encode(Y_test)

model = Network(layers=[784, 64, 10], activation_function='relu')

batch_size = 64
epochs = 300
learning_rate = 0.01
output = None
loss_history = []
m = X.shape[1]

for epoch in range(epochs):
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i:i + batch_size]
        Y_batch = Y_shuffled[:, i:i + batch_size]

        if X_batch.shape[1] == 0:
            continue

        output = model.train(X_batch, Y_batch, learning_rate)

    if epoch % 100 == 0:
        predictions = model.forward(X)
        loss = cross_entropy_loss(predictions, Y)
        loss_history.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

predictions = model.forward(X_test)
pred_labels = np.argmax(predictions, axis=0)
true_labels = np.argmax(Y_test_encoded, axis=0)

def show_prediction(image, true_label, pred_label):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f" Prediction: {pred_label}  |  Real: {true_label}")
    plt.axis('off')
    plt.show()

predictions = model.forward(X_test)
pred_labels = np.argmax(predictions, axis=0)
true_labels = np.argmax(Y_test_encoded, axis=0)

N = 20

for i in range(N):
    pred = pred_labels[i]
    real = true_labels[i]
    status = "✅" if pred == real else "❌"
    print(f"🖼 Imagen {i:02d} → {status} Real: {real} | Predicho: {pred}")


