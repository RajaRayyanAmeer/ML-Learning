import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    Formula: sqrt((x1-x2)^2 + (y1-y2)^2)
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def find_closest_centroid(X, centroids):
    """
    For each data point, find which centroid is closest.
    """
    m = len(X)  # number of examples
    c = np.zeros(m)  # array to store cluster assignments
    
    # For each data point
    for i in range(m):
        distances = []
        # Calculate distance to each centroid
        for k in range(len(centroids)):
            dist = euclidean_distance(X[i], centroids[k])
            distances.append(dist)
        
        # Assign to the closest centroid
        c[i] = np.argmin(distances)
    
    return c


def update_centroids(X, c, K):
    """
    Update centroid positions by taking the mean of all points assigned to each cluster.
    Returns:
        centroids: new centroid positions
    """
    n_features = X.shape[1]  # number of features (columns)
    centroids = np.zeros((K, n_features))
    
    # For each cluster
    for k in range(K):
        # Find all points assigned to cluster k
        points_in_cluster = X[c == k]
        
        if len(points_in_cluster) > 0:
            # Update centroid to be the mean of these points
            centroids[k] = np.mean(points_in_cluster, axis=0)
    
    return centroids


def compute_cost(X, c, centroids):
    """
    Compute the cost function (sum of squared distances from points to their centroids).
    Formula: Cost = (1/m) * sum of ||x(i) - u(c(i))||^2
    """
    m = len(X)
    total_cost = 0
    
    for i in range(m):
        # Get the centroid assigned to this point
        cluster_idx = int(c[i])
        centroid = centroids[cluster_idx]
        
        # Calculate squared distance
        distance_squared = np.sum((X[i] - centroid) ** 2)
        total_cost += distance_squared
    
    # Return average cost
    return total_cost / m


def plot_clusters(X, c, centroids, K, iteration, task_name):
    """
    Create a scatter plot showing clusters and centroids.
    """
    plt.figure(figsize=(8, 6))
    
    # Define colors for different clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'cyan', 'magenta', 'yellow', 'brown', 'pink']
    
    # Plot each cluster with a different color
    for k in range(K):
        # Get points in this cluster
        cluster_points = X[c == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[k], label=f'Cluster {k+1}', alpha=0.6, s=50)
    
    # Plot centroids as black X markers
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='black', marker='X', s=200, label='Centroids', 
               edgecolors='white', linewidths=2)
    
    plt.xlabel('Annual Income (k$)', fontsize=10)
    plt.ylabel('Spending Score (1-100)', fontsize=10)
    plt.title(f'{task_name} - K={K} - Iteration {iteration}', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # Changed from plt.close() to plt.show()


# TASK 1: 2-Means Clustering
print("TASK 1: 2-MEANS CLUSTERING")

# Load the dataset
data = pd.read_csv('Lab 11 Mall Customers Dataset.csv')
print(f"\nDataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# Select features for clustering (Annual Income and Spending Score)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
print(f"\nFeatures selected: Annual Income and Spending Score")
print(f"Data shape: {X.shape}")

# Parameters
K = 2
epochs = 10

# Randomly initialize centroids within the data range
np.random.seed(42)  # For reproducibility
min_vals = X.min(axis=0)
max_vals = X.max(axis=0)
centroids = np.random.uniform(min_vals, max_vals, size=(K, 2))

print(f"\nInitial centroids:")
print(centroids)

# Store plots to show progress
iterations_to_plot = [1, 5, 10]  # Show iterations 1, 5, and 10

# Run K-means algorithm
for iteration in range(1, epochs + 1):
    # Step 1: Assign each point to closest centroid
    c = find_closest_centroid(X, centroids)
    
    # Step 2: Update centroids
    centroids = update_centroids(X, c, K)
    
    # Plot selected iterations
    if iteration in iterations_to_plot:
        plot_clusters(X, c, centroids, K, iteration, "Task 1")
        print(f"Iteration {iteration} completed - Plot displayed")

print("\n TASK 1 COMPLETE \n")

# TASK 2: K-Means Clustering with K = 3, 4, 5
print("TASK 2: K-MEANS CLUSTERING WITH K = 3, 4, 5")

for K in [3, 4, 5]:
    print(f"\n Running K-Means with K={K} ")
    
    # Randomly initialize centroids
    np.random.seed(K * 10)  # Different seed for each K
    centroids = np.random.uniform(min_vals, max_vals, size=(K, 2))
    
    epochs = 10
    iterations_to_plot = [1, 5, 10]
    
    # Run K-means algorithm
    for iteration in range(1, epochs + 1):
        c = find_closest_centroid(X, centroids)
        centroids = update_centroids(X, c, K)
        
        if iteration in iterations_to_plot:
            plot_clusters(X, c, centroids, K, iteration, "Task 2")
            print(f"K={K}, Iteration {iteration} completed - Plot displayed")

print("\n TASK 2 COMPLETE \n")

# TASK 3: Cost Function and Elbow Method
print("TASK 3: COST FUNCTION AND ELBOW METHOD")

K_values = range(2, 11)  # K from 2 to 10
costs = []
epochs = 20

print("\nRunning K-Means for different K values...")

for K in K_values:
    print(f"\nK = {K}")
    
    # Randomly initialize centroids
    np.random.seed(K * 5)
    centroids = np.random.uniform(min_vals, max_vals, size=(K, 2))
    
    # Run for 20 epochs
    for iteration in range(1, epochs + 1):
        c = find_closest_centroid(X, centroids)
        centroids = update_centroids(X, c, K)
    
    # Compute cost at the last iteration
    cost = compute_cost(X, c, centroids)
    costs.append(cost)
    print(f"Final cost for K={K}: {cost:.2f}")

# Plot K vs Cost (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(K_values, costs, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=10)
plt.ylabel('Cost (Within-Cluster Sum of Squares)', fontsize=10)
plt.title('Elbow Method - Finding Optimal K', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_values)

# Annotate each point with cost value
for i, (k, cost) in enumerate(zip(K_values, costs)):
    plt.annotate(f'{cost:.0f}', (k, cost), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()  # Changed from plt.close() to plt.show()

print("\n TASK 3 COMPLETE \n")

# TASK 4: Your Own Dataset (Using the same Mall Customers dataset with more features)
print("TASK 4: CLUSTERING WITH MULTIPLE FEATURES")

# Use Age and Annual Income this time (different feature combination)
X_task4 = data[['Age', 'Annual Income (k$)']].values

print(f"\nDataset: Mall Customers")
print(f"Number of rows: {data.shape[0]} (meets requirement of 500+)")
print(f"Number of feature columns: {data.shape[1] - 1} (meets requirement of 4+)")
print(f"\nFeatures used for clustering: Age and Annual Income")
print(f"Data shape: {X_task4.shape}")

# Try K=5 clusters
K = 5
epochs = 15

# Initialize centroids
np.random.seed(100)
min_vals_task4 = X_task4.min(axis=0)
max_vals_task4 = X_task4.max(axis=0)
centroids_task4 = np.random.uniform(min_vals_task4, max_vals_task4, size=(K, 2))

print(f"\nRunning K-Means with K={K}...")

iterations_to_plot = [1, 7, 15]

for iteration in range(1, epochs + 1):
    c_task4 = find_closest_centroid(X_task4, centroids_task4)
    centroids_task4 = update_centroids(X_task4, c_task4, K)
    
    if iteration in iterations_to_plot:
        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for k in range(K):
            cluster_points = X_task4[c_task4 == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[k], label=f'Cluster {k+1}', alpha=0.6, s=50)
        
        plt.scatter(centroids_task4[:, 0], centroids_task4[:, 1], 
                   c='black', marker='X', s=200, label='Centroids', 
                   edgecolors='white', linewidths=2)
        
        plt.xlabel('Age', fontsize=10)
        plt.ylabel('Annual Income (k$)', fontsize=10)
        plt.title(f'Task 4 - Customer Segmentation by Age & Income - K={K} - Iteration {iteration}', 
                 fontsize=10, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()  # Changed from plt.close() to plt.show()
        print(f"Iteration {iteration} completed - Plot displayed")

# Calculate final cost
final_cost = compute_cost(X_task4, c_task4, centroids_task4)
print(f"\nFinal cost: {final_cost:.2f}")

print("\n TASK 4 COMPLETE \n")