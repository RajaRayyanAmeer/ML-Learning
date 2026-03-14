import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("METHOD 1: ")
df = pd.read_csv("House Price Univariate Dataset.csv")
x = df["Size_sqft"].values
y = df["Price"].values

x_mean = np.mean(x)
y_mean = np.mean(y)
m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
c = y_mean - (m * x_mean)
print("Slope: ",m)
print("Intercept: ",c)

y_pred = (m * x) + c
mse = np.mean((y - y_pred)**2)
print("MSE: ",mse)

ss_error = np.sum((y - y_pred)**2)
ss_total = np.sum((y - y_mean)**2)
r2 = 1 - (ss_error / ss_total)
print("R2: ",r2)

# Plot 1: Regression Line (Method 1)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Actual Data')
plt.plot(x, y_pred, color="red", linewidth=2, label='Regression Line')
plt.title("Method 1: Regression Line (Manual Calculation)")
plt.xlabel("Size (sqft)")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 2: Residual Plot (Method 1)
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(x, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title("Method 1: Residual Plot (Manual Calculation)")
plt.xlabel("Size (sqft)")
plt.ylabel("Residuals")
plt.grid(True, alpha=0.3)
plt.show()

print("METHOD 2: ")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

x2 = df[["Size_sqft"]].values
y2 = df["Price"].values

x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred2 = model.predict(x_test)

print("Slope: ", model.coef_)
print("Intercept: ", model.intercept_)
print("MSE: ", mean_squared_error(y_test, y_pred2))
print("R2: ", r2_score(y_test, y_pred2))

sorted_idx = x_test.flatten().argsort()
x_test_sorted = x_test[sorted_idx]
y_pred2_sorted = y_pred2[sorted_idx]

# Plot 3: Regression Line (Method 2)
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, alpha=0.6, label='Test Data')
plt.plot(x_test_sorted, y_pred2_sorted, color="red", linewidth=2, label='Regression Line')
plt.title("Method 2: Regression Line (Scikit-learn - Test Data)")
plt.xlabel("Size (sqft)")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 4: Residual Plot (Method 2)
residuals2 = y_test - y_pred2
plt.figure(figsize=(10, 6))
plt.scatter(x_test, residuals2, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title("Method 2: Residual Plot (Scikit-learn - Test Data)")
plt.xlabel("Size (sqft)")
plt.ylabel("Residuals")
plt.grid(True, alpha=0.3)
plt.show()

comparison = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred2.flatten(),
    'Error': y_test.flatten() - y_pred2.flatten()
})
print(comparison)