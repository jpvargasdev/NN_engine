import numpy as np
import matplotlib.pyplot as plt
from modelCNN import NetworkCNN
from utils import one_hot_encode_cnn, cross_entropy_loss, plot_accuracy_over_time, plot_loss, show_kernels, show_predictions

# Cargar y preparar datos
train_data = np.loadtxt('./dataset/mnist_train.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('./dataset/mnist_test.csv', delimiter=',', skiprows=1)

# Normalización de imágenes
X = train_data[:, 1:] / 255.0
Y = train_data[:, 0].astype(int)
X_test = test_data[:, 1:] / 255.0
Y_test = test_data[:, 0].astype(int)

X = X[:256]
Y = Y[:256]

# Reshape de imágenes a 28x28
X = X.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Codificación one-hot (forma: (n_samples, 10))
Y = one_hot_encode_cnn(Y, num_classes=10)
Y_test = one_hot_encode_cnn(Y_test, num_classes=10)

# Kernels
# Create random kernels with He initialization
conv_kernels = [np.random.randn(3, 3) * np.sqrt(2. / 9) for _ in range(5)]

# After conv + pool: 13×13 per feature map
flattened_input_size = 13 * 13 * len(conv_kernels)  # 507

# Dense architecture: [507] → 64 → 10
dense_sizes = [64, 10]

# Create the model
model = NetworkCNN(conv_kernels=conv_kernels, dense_sizes=dense_sizes, input_size=flattened_input_size)

batch_size     = 64
epochs         = 20
learning_rate  = 0.01
loss_history   = []

n_samples = X.shape[0]

for epoch in range(epochs):
    permutation = np.random.permutation(n_samples)
    X_shuffled  = X[permutation]
    Y_shuffled  = Y[permutation]

    epoch_loss = 0

    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]

        for x_img, y_label in zip(X_batch, Y_batch):
            x_img = x_img                  # shape: (28, 28)
            y_label = y_label.reshape(-1, 1)  # shape: (10, 1)

            loss, _ = model.train(x_img, y_label, learning_rate)
            epoch_loss += loss

    avg_loss = epoch_loss / n_samples

    if epoch % 1 == 0:
        loss_history.append(avg_loss)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}")

    if avg_loss < 0.05:
        break

test_size = 20
pred_labels = []
true_labels = np.argmax(Y_test[:test_size], axis=1)

for i in range(test_size):
    x_img = X_test[i]                 # shape (28, 28)
    output = model.forward(x_img)    # shape (10, 1)
    pred = np.argmax(output)
    pred_labels.append(pred)

# Compare predictions
for i in range(test_size):
    pred = pred_labels[i]
    real = true_labels[i]
    status = "✅" if pred == real else "❌"
    print(f"🖼 Imagen {i:02d} → {status} Real: {real} | Predicho: {pred}")

accuracy = np.mean(np.array(pred_labels) == true_labels)
print(f"\n🎯 Precisión sobre {test_size} imágenes: {accuracy * 100:.2f}%")

show_predictions(X_test[:test_size], true_labels, pred_labels)
plot_loss(loss_history)
plot_accuracy_over_time(acc_history=loss_history)
show_kernels(model.kernels)