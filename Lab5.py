# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Step 2: Dataset Reading
data = pd.read_csv("Lab 5 Melbourne Housing Full Dataset.csv")
print("Unique Values:\n", data.nunique())

# Step 3: Dataset Features
print("Dataset Shape:\n", data.shape)
print("Missing Values in the Dataset:\n", data.isna().sum())

# Step 4: Handling Missing Values and Strings
data = data.dropna(subset=["Price"])

data["Distance"] = data["Distance"].fillna(data["Distance"].mean())
data["BuildingArea"] = data["BuildingArea"].fillna(data["BuildingArea"].mean())

#Encode the Suburb column into numeric dummy variables
data = pd.get_dummies(data, columns=["Suburb"], drop_first=True)

suburb_cols = [col for col in data.columns if col.startswith("Suburb_")]
x = data[['Rooms', 'Distance', 'BuildingArea'] + suburb_cols]
y = data['Price']

# Step 5: Splitting the Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Step 6: Model Training
#Using Linear Regression
model = LinearRegression().fit(x_train, y_train)
model_test  = model.score(x_test,  y_test)
model_train = model.score(x_train, y_train)

#Using Ridge,L2 Regularization
ridge_model = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_model.fit(x_train,y_train)
ridge_test = ridge_model.score(x_test,y_test)
ridge_train = ridge_model.score(x_train,y_train)

#Using Lasso,L1 Regularization
lasso_model = Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_model.fit(x_train,y_train)
lasso_test = lasso_model.score(x_test,y_test)
lasso_train = lasso_model.score(x_train,y_train)

print("\nModel R2 Scores ")
print(f"Linear Regression\nTrain: {model_train:.4f}  | Test: {model_test:.4f}")
print(f"Ridge Regression\nTrain: {ridge_train:.4f}  | Test: {ridge_test:.4f}")
print(f"Lasso Regression\nTrain: {lasso_train:.4f}  | Test: {lasso_test:.4f}")

#Step 7: Visualization
sns.set_style("whitegrid")
models = ["Linear", "Ridge", "Lasso"]
train_scores = [model_train, ridge_train, lasso_train]
test_scores  = [model_test,  ridge_test,  lasso_test]

bar_width = 0.3
index = np.arange(len(models))

plt.figure(figsize=(9, 5))
plt.bar(index, train_scores, width=bar_width, label="Train Score", color="royalblue", alpha=0.8)
plt.bar(index + bar_width, test_scores,  width=bar_width, label="Test Score", color="orange", alpha=0.8)
plt.xlabel("Models", fontsize=12)
plt.ylabel("R² Score", fontsize=12)
plt.title("Train vs Test R² Scores — Linear, Ridge & Lasso", fontsize=13)
plt.xticks(index + bar_width / 2, models)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()