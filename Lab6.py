# Step 1: Import Libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Step 2: Load Dataset
df = pd.read_csv("Lab 6 Insurance Dataset.csv")

# Step 3: Visualize Data
plt.scatter(df.age, df.bought_insurance, marker = '+', color = 'red')
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.title("Age vs Insurance Purchase")
plt.show()

# Step 4: Train and Evaluate Models for Multiple Split Ratios
split_ratios = [0.8, 0.7, 0.6]  
labels = ['80-20', '70-30', '60-40']
accuracies = []

print("\nModel Evaluation for Different Split Ratios")
for ratio, label in zip(split_ratios, labels):
    print(f"SPLIT RATIO: {label}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df[['age']], df.bought_insurance, train_size = ratio, random_state = 42
    )

    # Train Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_predicted = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("\nPredicted Classes:")
    print(y_predicted)

    print("\nPredicted Probabilities [P(No), P(Yes)]:")
    print(y_prob)

    # Model Accuracy
    accuracy = model.score(X_test, y_test)
    accuracies.append(accuracy)
    print(f"\nModel Accuracy ({label}): {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_predicted)
    print("\nConfusion Matrix:")
    print(cm)

# Bar graph: accuracy vs split ratio
plt.bar(labels, accuracies, color = ['steelblue', 'darkorange', 'seagreen'])
plt.xlabel("Train-Test Split Ratio")
plt.ylabel("Accuracy")
plt.title("Model Accuracy for Different Split Ratios")
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha = 'center', fontweight = 'bold')
plt.show()

# Step 5: Sigmoid Function Visualization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.grid()
plt.show()

# Step 6: Test on New Data
new_age = [[35]]

prediction = model.predict(new_age)
probability = model.predict_proba(new_age)

print("\nNew Input (Age = 35)")
print("Prediction:", prediction)
print("Probability [P(No), P(Yes)]:", probability)

# Step 7: Manual Prediction Function
def manual_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(age):
    z = 0.042 * age - 1.53
    return manual_sigmoid(z)

print("\nManual Predictions:")
print("Age 35:", prediction_function(35))
print("Age 43:", prediction_function(43))