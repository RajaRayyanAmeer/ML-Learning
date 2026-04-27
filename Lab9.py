import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TASK 1: Activation Functions
print(" TASK 1: Activation Functions \n")
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

z_test = np.array([[-1.0, 0.0, 1.0, 2.0]])
print(" Sigmoid :", sigmoid(z_test))
print(" ReLU :", relu(z_test))
print(" dReLU :", drelu(z_test))

# TASK 2: Matrix Initializations
print("\n TASK 2: Matrix Initializations \n")
df = pd.read_excel('Lab 8 + 9 Students Dataset.csv')

X = df.iloc[:, :-1].values.T   
Y = df.iloc[:, -1].values.reshape(1, -1) 

np.random.seed(42)           
m = X.shape[1]
indices = np.random.permutation(m)
train_size = int(m * 0.8)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

Xtrain, Ytrain = X[:, train_idx], Y[:, train_idx]
Xtest, Ytest   = X[:, test_idx],   Y[:, test_idx]

n_x = Xtrain.shape[0]              
m_train = Xtrain.shape[1]
m_test  = Xtest.shape[1]

n_h1 = 4   # neurons in hidden layer 1
n_h2 = 4   # neurons in hidden layer 2
n_y  = 1   # output neurons

Z1 = np.zeros((n_h1, m_train))
A1 = np.zeros((n_h1, m_train))
Z2 = np.zeros((n_h2, m_train))
A2 = np.zeros((n_h2, m_train))
Z3 = np.zeros((n_y, m_train))
A3 = np.zeros((n_y, m_train))

dZ1 = np.zeros_like(Z1)
dZ2 = np.zeros_like(Z2)
dZ3 = np.zeros_like(Z3)
dZ1cache = np.zeros_like(Z1)
dZ2cache = np.zeros_like(Z2)
dZ3cache = np.zeros_like(Z3)

np.random.seed(0)
W1 = np.random.randn(n_h1, n_x) * 0.01
B1 = np.zeros((n_h1, 1))
W2 = np.random.randn(n_h2, n_h1) * 0.01
B2 = np.zeros((n_h2, 1))
W3 = np.random.randn(n_y, n_h2) * 0.01
B3 = np.zeros((n_y, 1))

dW1 = np.zeros_like(W1)
dB1 = np.zeros_like(B1)
dW2 = np.zeros_like(W2)
dB2 = np.zeros_like(B2)
dW3 = np.zeros_like(W3)
dB3 = np.zeros_like(B3)

print("\nDataSet shapes:")
print(f"Xtrain: {Xtrain.shape}, Ytrain: {Ytrain.shape}")
print(f"Xtest:  {Xtest.shape},  Ytest:  {Ytest.shape}")
print(f"W1: {W1.shape}, B1: {B1.shape}")
print(f"W2: {W2.shape}, B2: {B2.shape}")
print(f"W3: {W3.shape}, B3: {B3.shape}")

# TASK 3: Forward Propagation
print("\n TASK 3: Forward Propagation \n")
def forward_propagation(X, Y, W1, B1, W2, B2, W3, B3):
    Z1 = W1 @ X + B1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + B2
    A2 = relu(Z2)
    Z3 = W3 @ A2 + B3
    A3 = sigmoid(Z3)

    m = X.shape[1]
    cost = - (1/m) * np.sum(Y * np.log(A3 + 1e-8) + (1-Y) * np.log(1 - A3 + 1e-8))
    return Z1, A1, Z2, A2, Z3, A3, cost

_, _, _, _, _, _, train_cost = forward_propagation(Xtrain, Ytrain, W1, B1, W2, B2, W3, B3)
_, _, _, _, _, _, test_cost  = forward_propagation(Xtest, Ytest, W1, B1, W2, B2, W3, B3)

print(f"\nInitial training cost: {train_cost:.4f}")
print(f" Initial test cost:     {test_cost:.4f}")

# TASK 4: Backward Propagation & Weight Update
print("\n TASK 4: Backward Propagation & Weight Update \n")
def backward_propagation(X, Y, Z1, A1, Z2, A2, A3, W2, W3):
    m = X.shape[1]

    dZ3cache = A3 - Y
    dZ3 = dZ3cache
    dW3 = (1/m) * dZ3 @ A2.T
    dB3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2cache = (W3.T @ dZ3) * drelu(Z2)
    dZ2 = dZ2cache
    dW2 = (1/m) * dZ2 @ A1.T
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1cache = (W2.T @ dZ2) * drelu(Z1)
    dZ1 = dZ1cache
    dW1 = (1/m) * dZ1 @ X.T
    dB1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, dB1, dW2, dB2, dW3, dB3, dZ1, dZ2, dZ3, dZ1cache, dZ2cache, dZ3cache

Z1, A1, Z2, A2, Z3, A3, cost = forward_propagation(Xtrain, Ytrain, W1, B1, W2, B2, W3, B3)
dW1, dB1, dW2, dB2, dW3, dB3, dZ1, dZ2, dZ3, dZ1c, dZ2c, dZ3c = backward_propagation(
    Xtrain, Ytrain, Z1, A1, Z2, A2, A3, W2, W3)

print("\n Initial weights (W1): \n", W1)
print("\n Initial biases (B1): \n", B1)
print("\n Training cost: ", cost)
print("\n Gradient for W1 (dW1): \n", dW1)
print("\n Gradient for B1 (dB1): \n", dB1)

alpha = 0.01
W1_updated = W1 - alpha * dW1
B1_updated = B1 - alpha * dB1
W2_updated = W2 - alpha * dW2
B2_updated = B2 - alpha * dB2
W3_updated = W3 - alpha * dW3
B3_updated = B3 - alpha * dB3

print("\n Updated W1 after one step: \n", W1_updated)

# TASK 5: Training and Testing Loop
print("\n TASK 5: Training and Testing Loop \n")
def train_neural_network(Xtrain, Ytrain, Xtest, Ytest,
                         W1, B1, W2, B2, W3, B3,
                         alpha, epochs):
    train_costs = []
    test_costs = []

    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3, train_cost = forward_propagation(
            Xtrain, Ytrain, W1, B1, W2, B2, W3, B3)
        dW1, dB1, dW2, dB2, dW3, dB3, _, _, _, _, _, _ = backward_propagation(
            Xtrain, Ytrain, Z1, A1, Z2, A2, A3, W2, W3)
        W1 -= alpha * dW1
        B1 -= alpha * dB1
        W2 -= alpha * dW2
        B2 -= alpha * dB2
        W3 -= alpha * dW3
        B3 -= alpha * dB3

        _, _, _, _, _, _, test_cost = forward_propagation(
            Xtest, Ytest, W1, B1, W2, B2, W3, B3)

        train_costs.append(train_cost)
        test_costs.append(test_cost)

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} Train cost: {train_cost:.4f}, Test cost: {test_cost:.4f}")

    return train_costs, test_costs, W1, B1, W2, B2, W3, B3

epochs = 1000
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]  

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
axes = axes.ravel()

for i, lr in enumerate(learning_rates):
    print(f"\nTraining with alpha = {lr}")

    np.random.seed(0)
    W1 = np.random.randn(n_h1, n_x) * 0.01
    B1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    B2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    B3 = np.zeros((n_y, 1))

    train_costs, test_costs, W1_final, B1_final, W2_final, B2_final, W3_final, B3_final = train_neural_network(
        Xtrain, Ytrain, Xtest, Ytest, W1, B1, W2, B2, W3, B3, lr, epochs)

    axes[i].plot(train_costs, label='Train')
    axes[i].plot(test_costs, label='Test')
    axes[i].set_title(f'α = {lr}')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Cost')
    axes[i].legend()
    axes[i].grid(True)

    if i == 0 or min(test_costs) < best_test_cost:
        best_test_cost = min(test_costs)
        best_weights = (W1_final, B1_final, W2_final, B2_final, W3_final, B3_final)

W1_best, B1_best, W2_best, B2_best, W3_best, B3_best = best_weights
np.save("W1_best.npy", W1_best)
np.save("B1_best.npy", B1_best)
np.save("W2_best.npy", W2_best)
np.save("B2_best.npy", B2_best)
np.save("W3_best.npy", W3_best)
np.save("B3_best.npy", B3_best)
print("\nBest weights saved (lowest test cost).")

plt.tight_layout()
plt.show()