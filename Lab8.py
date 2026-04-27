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