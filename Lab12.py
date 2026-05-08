import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

heart_df = pd.read_csv("Lab 12 Heart Disease UCI.csv")
student_df = pd.read_csv("Lab 12 Student Data.csv")

# Keep only numeric data
heart_df = heart_df.select_dtypes(include=[np.number]).dropna()
student_df = student_df.select_dtypes(include=[np.number]).dropna()

# TASK 1 – FEATURE NORMALIZATION
print("\n TASK 1: NORMALIZATION ")

def normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_heart = heart_df.values
X_norm, mu, sigma = normalize(X_heart)

print("Mean:\n", mu)
print("Std Dev:\n", sigma)
print("Normalized Sample:\n", X_norm[:5])

# TASK 2 – COVARIANCE MATRIX
print("\n TASK 2: COVARIANCE MATRIX ")

m = X_norm.shape[0]


cov_matrix = (X_norm.T @ X_norm) / m

print("Covariance (Matrix):\n", cov_matrix)

# TASK 3 – PCA (EIGEN DECOMPOSITION)
print("\n TASK 3: PCA ")

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Reduce dimensions (k=2)
k = 2
U_reduced = eigenvectors[:, :k]

# Project data
Z = X_norm @ U_reduced

print("Top Eigenvalues:\n", eigenvalues[:k])
print("Projected Data Sample:\n", Z[:5])

# TASK 4 – PROJECTION PLOTTING
print("\n TASK 4: PLOTTING ")

# Create synthetic 3D planar data
np.random.seed(0)
X3D = np.random.rand(200, 2)
Z_plane = X3D[:, 0] + X3D[:, 1]
X3D = np.column_stack((X3D, Z_plane))

# Normalize
X3D_norm, _, _ = normalize(X3D)

# PCA to 2D
cov_3d = (X3D_norm.T @ X3D_norm) / len(X3D_norm)
eigval_3d, eigvec_3d = np.linalg.eig(cov_3d)
idx3d = np.argsort(eigval_3d)[::-1]
U2 = eigvec_3d[:, idx3d[:2]]

Z2D = X3D_norm @ U2

# Plot
fig = plt.figure(figsize=(10,4))

# 3D plot
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X3D_norm[:,0], X3D_norm[:,1], X3D_norm[:,2])
ax.set_title("Original 3D Data")

# 2D projection
plt.subplot(122)
plt.scatter(Z2D[:,0], Z2D[:,1])
plt.title("Projected 2D Data")

plt.show()

# TASK 5 – PCA + LINEAR REGRESSION
print("\n TASK 5: PCA + REGRESSION ")

# Use student dataset
X = student_df.iloc[:, :-1].values
y = student_df.iloc[:, -1].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# --- WITHOUT PCA ---
model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_loss = mean_squared_error(y_train, train_pred)
val_loss = mean_squared_error(y_val, val_pred)

print("Without PCA, Train Loss: ", train_loss)
print("Without PCA, Val Loss: ", val_loss)

# --- WITH PCA ---
X_norm, _, _ = normalize(X)

cov = (X_norm.T @ X_norm) / len(X_norm)
eigval, eigvec = np.linalg.eig(cov)

idx = np.argsort(eigval)[::-1]
U_k = eigvec[:, idx[:2]]

Z = X_norm @ U_k

Z_train, Z_val, y_train, y_val = train_test_split(Z, y, test_size=0.2)

model_pca = LinearRegression()
model_pca.fit(Z_train, y_train)

train_pred_pca = model_pca.predict(Z_train)
val_pred_pca = model_pca.predict(Z_val)

train_loss_pca = mean_squared_error(y_train, train_pred_pca)
val_loss_pca = mean_squared_error(y_val, val_pred_pca)

print("With PCA, Train Loss: ", train_loss_pca)
print("With PCA, Val Loss: ", val_loss_pca)

# BAR GRAPH COMPARISON

labels = ['Train Loss', 'Validation Loss']

without_pca = [train_loss, val_loss]
with_pca = [train_loss_pca, val_loss_pca]

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, without_pca, width, label='Without PCA')
plt.bar(x + width/2, with_pca, width, label='With PCA')

plt.xticks(x, labels)
plt.ylabel("Loss")
plt.title("Loss Comparison (With vs Without PCA)")
plt.legend()
plt.show()