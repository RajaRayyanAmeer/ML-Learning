import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Task 1: Gaussian Distribution")

def gaussian(x, mean, std):
    """Return the Gaussian probability density."""
    coeff = 1 / (std * np.sqrt(2 * np.pi))
    exp = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return coeff * exp

# Generate x values
x = np.linspace(-15, 15, 1000)

# Three different distributions
y1 = gaussian(x, mean=0, std=1)
y2 = gaussian(x, mean=5, std=2)
y3 = gaussian(x, mean=-4, std=1.5)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='μ=0, σ=1')
plt.plot(x, y2, label='μ=5, σ=2')
plt.plot(x, y3, label='μ=-4, σ=1.5')
plt.title('Gaussian Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

print("Task 2: Dataset with Outliers")
# Load dataset
data = pd.read_csv("Lab 13 Credit Card Fraud Dataset.csv")

# Use first 500 rows and three numeric features
df = data[['TransactionID', 'Amount', 'MerchantID']].head(500).copy()

# Scatter plot before adding outliers (using TransactionID vs Amount)
plt.figure(figsize=(10, 6))
plt.scatter(df['TransactionID'], df['Amount'], alpha=0.6)
plt.title('Scatter Plot – Original Data')
plt.xlabel('TransactionID')
plt.ylabel('Amount')
plt.grid(True)
plt.show()

# Mean and standard deviation of each feature
print("Mean of each feature:")
print(df.mean())
print("\nStandard deviation of each feature:")
print(df.std())

# Create three obvious outliers
outliers = pd.DataFrame({
    'TransactionID': [10000, 12000, 15000],
    'Amount': [10000, 20000, 30000],
    'MerchantID': [9999, 9999, 9999]
})

# Add outliers to the dataset
df_out = pd.concat([df, outliers], ignore_index=True)

# Scatter plot after adding outliers
plt.figure(figsize=(10, 6))
plt.scatter(df_out['TransactionID'], df_out['Amount'], alpha=0.6)
plt.scatter(outliers['TransactionID'], outliers['Amount'], 
            color='red', s=100, label='Added outliers')
plt.title('Scatter Plot – After Adding Outliers')
plt.xlabel('TransactionID')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.show()

print("Task 3: Anomaly Detection")
# Use the dataset with outliers (or original plus outliers)
df_detect = df_out.copy()

# Calculate mean and std for each feature
mean_id = df_detect['TransactionID'].mean()
std_id = df_detect['TransactionID'].std()
mean_amt = df_detect['Amount'].mean()
std_amt = df_detect['Amount'].std()
mean_mer = df_detect['MerchantID'].mean()
std_mer = df_detect['MerchantID'].std()

# Compute Gaussian probability for each feature
prob_id = gaussian(df_detect['TransactionID'], mean_id, std_id)
prob_amt = gaussian(df_detect['Amount'], mean_amt, std_amt)
prob_mer = gaussian(df_detect['MerchantID'], mean_mer, std_mer)

# Combined probability (product – assumes independence)
prob_total = prob_id * prob_amt * prob_mer

# Define threshold (e.g., 1st percentile)
threshold = np.percentile(prob_total, 1)

# Flag anomalies
anomalies = df_detect[prob_total < threshold]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df_detect['TransactionID'], df_detect['Amount'], 
            c='blue', alpha=0.5, label='Normal')
plt.scatter(anomalies['TransactionID'], anomalies['Amount'], 
            c='red', s=100, label='Detected Anomalies')
plt.title('Anomaly Detection using Gaussian Distribution')
plt.xlabel('TransactionID')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.show()

print(f"Number of detected anomalies: {len(anomalies)}")
print(anomalies)