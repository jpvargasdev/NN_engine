# NN_engine


# 🧠 Neural Network Cheat Sheet

---

## ⚙️ 1. Forward Pass

**Pre-activation (input to neuron):**
```
z^[l] = W^[l] · a^[l-1] + b^[l]
```

**Activation (output of neuron):**
```
a^[l] = σ(z^[l])
```

Where:
- `W^[l]`: weight matrix of layer l
- `b^[l]`: bias vector of layer l
- `a^[l-1]`: activations from previous layer
- `σ`: activation function (commonly sigmoid, tanh, ReLU, etc.)

---

## 💥 2. Loss Function (MSE)

**Mean Squared Error:**
```
L = (1/n) ∑ (ŷᵢ - yᵢ)²
```

**Gradient of Loss w.r.t. prediction (output):**
```
∂L/∂ŷ = 2(ŷ - y)
```

---

## 🔁 3. Backpropagation

**Output layer error:**
```
δ^[L] = (ŷ - y) ⊙ σ'(z^[L])
```

**Hidden layer error (general case):**
```
δ^[l] = (W^[l+1])ᵗ · δ^[l+1] ⊙ σ'(z^[l])
```

Where:
- `δ^[l]`: error in layer l
- `⊙`: element-wise multiplication (Hadamard product)
- `σ'(z)`: derivative of activation function

For sigmoid:
```
σ'(z) = σ(z)(1 - σ(z)) = a^[l](1 - a^[l])
```

---

## 📉 4. Gradient Descent Updates

**Update rule for weights:**
```
W^[l] := W^[l] - η ∂L/∂W^[l]
```

**Update rule for biases:**
```
b^[l] := b^[l] - η ∂L/∂b^[l]
```

Where:
- `η`: learning rate

**Gradients:**
```
∂L/∂W^[l] = δ^[l] · (a^[l-1])ᵗ
∂L/∂b^[l] = δ^[l]
```

---

## 🔢 Dimensions Recap

- `W^[l]`: shape = [neurons in layer l, neurons in layer l-1]
- `b^[l]`: shape = [neurons in layer l, 1]
- `a^[l]`, `δ^[l]`: shape = [neurons in layer l, 1]

