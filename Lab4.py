#Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

#Step 2: Import the Dataset, and Reading
data = pd.read_csv("Lab 4 Laptop Prices Multivariate Dataset.csv")

#Step 3: Data Preprocessing
#Checking for the Missing Values
print("\nMissing Values in the Dataset: \n",data.isnull().sum())

#Handling the Missing Values
#Dependent Values Missing (Target One), Drop Rows
data = data.dropna(subset=["Price_USD"])

#Independent Values (Replacing the Missing Values with Column Means, if there is no Outlier. But if there is any outliers, we will go with the Median Approach)
data["RAM_GB"] = data["RAM_GB"].fillna(data["RAM_GB"].mean())
data["Processor_Speed_GHz"] = data["Processor_Speed_GHz"].fillna(data["Processor_Speed_GHz"].mean())

#Independent Variables
x = data[['RAM_GB','Storage_GB','Processor_Speed_GHz','Screen_Size_Inches']]

#Dependent Variables
y = data['Price_USD']

#Step 4: Train_Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Step 5: Model Training
model = LinearRegression()
model.fit(x_train,y_train)

print("Intercept (c): ",model.intercept_)
print("Slope (m1, m2, m3, m4 respectively): ",model.coef_)

#Step 6: Predicting
y_pred = model.predict(x_test)
comparison = pd.DataFrame({
    'Actual':y_test.values,
    'Predicted':y_pred})
print(comparison)

#Step 7: Model Evaluation
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("\n Model Evaluation Metrics \n")
print("Mean Squared Error (MSE): ",mse)
print("R^2 Score: ",r2)

#Step 8: Data Visulization
#Plot Actual vs Predicted Values
plt.figure(figsize = (8,6))
sns.scatterplot(x = y_test, y = y_pred, color = "blue")
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(), y_test.max()],
         color = "red",
         linestyle = "--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

#Plot Regression Line
plt.figure(figsize = (8,6))
sns.scatterplot(x = y_pred, y = y_test, color = "blue")
plt.plot([y_pred.min(), y_pred.max()],
         [y_pred.min(), y_pred.max()],
         color = "red",
         linestyle = "-")
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.title("Regression Line - Predicted vs Actual")
plt.show()

#Plot Residual Error
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, color="blue")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual Error")
plt.title("Residual Plot")
plt.show()