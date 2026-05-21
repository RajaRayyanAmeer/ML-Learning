import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# DATA LOADING AND EXPLORATION 
print("TASK 1: DATA LOADING AND EXPLORATION")

# Try different encodings for reading CSV files
encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']

movies = None
ratings = None

# Try loading movies.csv with different encodings
for encoding in encodings_to_try:
    try:
        movies = pd.read_csv('movies.csv', encoding=encoding)
        print(f"Successfully loaded movies.csv with {encoding} encoding")
        break
    except (UnicodeDecodeError, UnicodeError):
        continue

if movies is None:
    movies = pd.read_csv('movies.csv', encoding='utf-8', errors='ignore')
    print("Loaded movies.csv with utf-8 and error ignoring")

# Try loading ratings.csv with different encodings
for encoding in encodings_to_try:
    try:
        ratings = pd.read_csv('ratings.csv', encoding=encoding)
        print(f"Successfully loaded ratings.csv with {encoding} encoding")
        break
    except (UnicodeDecodeError, UnicodeError):
        continue

if ratings is None:
    ratings = pd.read_csv('ratings.csv', encoding='utf-8', errors='ignore')
    print("Loaded ratings.csv with utf-8 and error ignoring")

print(f"\nRatings dataset shape: {ratings.shape}")
print(f"Movies dataset shape: {movies.shape}")

# Total number of unique users and movies
n_users = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
print(f"\nTotal unique users: {n_users}")
print(f"Total unique movies: {n_movies}")

# Plot distribution of ratings
plt.figure(figsize=(10, 6))
ratings['rating'].hist(bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Construct user-item ratings matrix
user_item_matrix = ratings.pivot_table(
    index='userId', 
    columns='movieId', 
    values='rating'
).fillna(0)

print(f"\nUser-Item matrix shape: {user_item_matrix.shape}")

# Calculate sparsity
n_zero = (user_item_matrix == 0).sum().sum()
total_entries = user_item_matrix.shape[0] * user_item_matrix.shape[1]
sparsity = 1 - (total_entries - n_zero) / total_entries
print(f"Matrix sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")

# CONTENT-BASED FILTERING
print("\nTASK 2: USER-BASED FILTERING")

# Prepare genre features
movies['genres'] = movies['genres'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity between movies
movie_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def content_recommendations(title, n=5):
    """Get content-based recommendations for a movie"""
    if title not in indices.index:
        return f"Movie '{title}' not found in dataset"
    
    idx = indices[title]
    sim_scores = list(enumerate(movie_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    recommendations = []
    for i in movie_indices:
        recommendations.append((movies['title'].iloc[i], float(sim_scores[movie_indices.index(i)][1])))
    
    return recommendations

def build_user_profile(user_id, ratings_df, movies_df, tfidf_matrix):
    """Build user profile based on highly rated movies (>=4 stars)"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    high_rated = user_ratings[user_ratings['rating'] >= 4]
    
    if len(high_rated) == 0:
        return None
    
    # Get feature vectors for highly rated movies
    movie_indices = []
    for movie_id in high_rated['movieId']:
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if len(movie_row) > 0:
            idx = movie_row.index[0]
            movie_indices.append(idx)
    
    if not movie_indices:
        return None
    
    # Average the feature vectors and convert to array (FIX APPLIED HERE)
    user_profile = np.asarray(tfidf_matrix[movie_indices].mean(axis=0))
    return user_profile

def recommend_for_user(user_id, ratings_df, movies_df, tfidf_matrix, n=5):
    """Recommend movies for a user based on their profile"""
    user_profile = build_user_profile(user_id, ratings_df, movies_df, tfidf_matrix)
    
    if user_profile is None:
        return "User has no highly rated movies"
    
    # Get movies the user has already rated
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    
    # Calculate similarity between user profile and all movies
    similarities = cosine_similarity(user_profile, tfidf_matrix)[0]
    
    # Create list of (movie_index, similarity) for unrated movies
    unrated_indices = []
    for idx, row in movies_df.iterrows():
        if row['movieId'] not in rated_movies:
            unrated_indices.append((idx, similarities[idx]))
    
    # Sort by similarity and get top n
    unrated_indices.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in unrated_indices[:n]]
    
    recommendations = []
    for idx in top_indices:
        recommendations.append((movies_df['title'].iloc[idx], float(similarities[idx])))
    
    return recommendations

# Test content-based recommendations
test_movie = "Toy Story (1995)"
if test_movie in indices.index:
    print(f"\nContent-based recommendations for '{test_movie}':")
    content_recs = content_recommendations(test_movie, 5)
    if isinstance(content_recs, list):
        for i, (movie, score) in enumerate(content_recs, 1):
            print(f"{i}. {movie} -> Similarity: {score:.4f}")
    else:
        print(content_recs)
else:
    print(f"\n'{test_movie}' not found. Trying alternative movie...")
    available_movies = movies['title'].head(10).tolist()
    print(f"Available movies: {available_movies[:5]}")
    test_movie = available_movies[0]
    print(f"\nContent-based recommendations for '{test_movie}':")
    content_recs = content_recommendations(test_movie, 5)
    if isinstance(content_recs, list):
        for i, (movie, score) in enumerate(content_recs, 1):
            print(f"{i}. {movie} -> Similarity: {score:.4f}")

# Test user-based content recommendations
if len(ratings['userId'].unique()) > 0:
    test_user = ratings['userId'].iloc[0]
    print(f"\nContent-based recommendations for User {test_user}:")
    user_content_recs = recommend_for_user(test_user, ratings, movies, tfidf_matrix, 5)
    if isinstance(user_content_recs, list):
        for i, (movie, score) in enumerate(user_content_recs, 1):
            print(f"{i}. {movie} -> Score: {score:.4f}")
    else:
        print(user_content_recs)

# COLLABORATIVE FILTERING
print("\nTASK 3: ITEM-BASED COLLABORATIVE FILTERING")

def mean_center_matrix(matrix):
    """Mean-center the ratings matrix (subtract user mean from non-zero entries)"""
    mean_centered = matrix.copy()
    user_means = matrix.mean(axis=1)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                mean_centered[i, j] = matrix[i, j] - user_means[i]
    
    return mean_centered, user_means

def cosine_similarity_users(matrix):
    """Compute cosine similarity between all users"""
    n_users = matrix.shape[0]
    similarity = np.zeros((n_users, n_users))
    
    for i in range(n_users):
        for j in range(n_users):
            # Get non-zero ratings for both users
            mask_i = matrix[i, :] != 0
            mask_j = matrix[j, :] != 0
            common_mask = mask_i & mask_j
            
            if np.sum(common_mask) > 0:
                vec_i = matrix[i, common_mask]
                vec_j = matrix[j, common_mask]
                
                # Cosine similarity
                dot_product = np.dot(vec_i, vec_j)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity[i, j] = dot_product / (norm_i * norm_j)
    
    return similarity

def predict_rating_collab(user_id, movie_id, matrix, similarity, user_means, k=10):
    """Predict rating using user-based collaborative filtering"""
    user_ids = np.unique(np.where(matrix > 0)[0])
    movie_ids = np.unique(np.where(matrix > 0)[1])
    
    if user_id not in user_ids or movie_id not in movie_ids:
        return user_means[user_id] if user_id < len(user_means) else 3.0
    
    user_idx = np.where(user_ids == user_id)[0][0] if user_id in user_ids else -1
    movie_idx = np.where(movie_ids == movie_id)[0][0] if movie_id in movie_ids else -1
    
    if user_idx == -1 or movie_idx == -1:
        return user_means[user_id] if user_id < len(user_means) else 3.0
    
    if matrix[user_idx, movie_idx] != 0:
        return matrix[user_idx, movie_idx]
    
    # Find similar users who have rated this movie
    similar_users = []
    for other_user in range(matrix.shape[0]):
        if other_user != user_idx and matrix[other_user, movie_idx] != 0:
            sim = similarity[user_idx, other_user]
            if sim > 0:
                similar_users.append((other_user, sim))
    
    similar_users.sort(key=lambda x: x[1], reverse=True)
    similar_users = similar_users[:k]
    
    if len(similar_users) == 0:
        return user_means[user_idx]
    
    # Weighted average prediction
    numerator = 0
    denominator = 0
    for other_user, sim in similar_users:
        rating = matrix[other_user, movie_idx]
        adjusted_rating = rating - user_means[other_user]
        numerator += sim * adjusted_rating
        denominator += abs(sim)
    
    if denominator > 0:
        prediction = user_means[user_idx] + (numerator / denominator)
    else:
        prediction = user_means[user_idx]
    
    prediction = max(0.5, min(5.0, prediction))
    return prediction

# Prepare data for collaborative filtering
user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()

valid_users = user_counts[user_counts >= 10].index
valid_movies = movie_counts[movie_counts >= 5].index

filtered_ratings = ratings[
    ratings['userId'].isin(valid_users) & 
    ratings['movieId'].isin(valid_movies)
]

# Create smaller matrix for demonstration
n_users_sample = min(100, len(valid_users))
n_movies_sample = min(200, len(valid_movies))

sample_users = valid_users[:n_users_sample] if len(valid_users) > 0 else ratings['userId'].unique()[:100]
sample_movies = valid_movies[:n_movies_sample] if len(valid_movies) > 0 else ratings['movieId'].unique()[:200]

sample_ratings = filtered_ratings[
    filtered_ratings['userId'].isin(sample_users) & 
    filtered_ratings['movieId'].isin(sample_movies)
]

if len(sample_ratings) > 0:
    collab_matrix = sample_ratings.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0).values
    
    mean_centered_matrix, user_means = mean_center_matrix(collab_matrix)
    user_similarity = cosine_similarity_users(mean_centered_matrix)
    
    if len(collab_matrix) > 0 and len(sample_users) > 0 and len(sample_movies) > 0:
        test_user = sample_users[0]
        test_movie_id = sample_movies[0]
        
        print(f"\nPredicting rating for User {test_user}, Movie {test_movie_id}:")
        collab_pred = predict_rating_collab(
            test_user, test_movie_id, collab_matrix, 
            user_similarity, user_means, k=10
        )
        print(f"User-based CF Prediction: {collab_pred:.2f}")
else:
    print("\nNot enough data for collaborative filtering. Using sample data instead.")
    sample_size = min(50, len(ratings['userId'].unique()))
    sample_users = ratings['userId'].unique()[:sample_size]
    sample_movies = ratings['movieId'].unique()[:100]
    
    sample_ratings = ratings[
        ratings['userId'].isin(sample_users) & 
        ratings['movieId'].isin(sample_movies)
    ]
    
    if len(sample_ratings) > 0:
        collab_matrix = sample_ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0).values
        
        mean_centered_matrix, user_means = mean_center_matrix(collab_matrix)
        user_similarity = cosine_similarity_users(mean_centered_matrix)
        
        test_user = sample_users[0]
        test_movie_id = sample_movies[0]
        
        print(f"\nPredicting rating for User {test_user}, Movie {test_movie_id}:")
        collab_pred = predict_rating_collab(
            test_user, test_movie_id, collab_matrix, 
            user_similarity, user_means, k=10
        )
        print(f"User-based CF Prediction: {collab_pred:.2f}")

# HYBRID RECOMMENDER
print("\nTASK 4: CONTENT-BASED FILTERING")

def hybrid_recommendation(user_id, movie_title, alpha=0.6):
    """Combine content-based and collaborative filtering"""
    content_recs = content_recommendations(movie_title, 10)
    
    if not isinstance(content_recs, list):
        return []
    
    hybrid_results = []
    
    for rec_movie, content_score in content_recs:
        movie_row = movies[movies['title'] == rec_movie]
        if len(movie_row) > 0:
            movie_id = movie_row['movieId'].iloc[0]
            
            try:
                collab_score = predict_rating_collab(
                    user_id, movie_id, collab_matrix,
                    user_similarity, user_means, k=10
                )
                collab_score_norm = collab_score / 5.0
            except:
                collab_score_norm = 0.5
            
            hybrid_score = alpha * content_score + (1 - alpha) * collab_score_norm
            hybrid_results.append((rec_movie, hybrid_score))
    
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    return hybrid_results[:5]

# Test hybrid recommendation
if 'collab_matrix' in locals() and len(collab_matrix) > 0:
    test_user = sample_users[0] if len(sample_users) > 0 else 1
    test_movie = "Toy Story (1995)"
    
    if test_movie in indices.index:
        print(f"\nHybrid recommendations for User {test_user} based on '{test_movie}':")
        hybrid_recs = hybrid_recommendation(test_user, test_movie, alpha=0.6)
        for i, (movie, score) in enumerate(hybrid_recs, 1):
            print(f"{i}. {movie} -> Hybrid Score: {score:.4f}")
    else:
        test_movie = movies['title'].iloc[0]
        print(f"\nHybrid recommendations for User {test_user} based on '{test_movie}':")
        hybrid_recs = hybrid_recommendation(test_user, test_movie, alpha=0.6)
        if hybrid_recs:
            for i, (movie, score) in enumerate(hybrid_recs, 1):
                print(f"{i}. {movie} -> Hybrid Score: {score:.4f}")
else:
    print("\nSkipping hybrid recommendations due to insufficient collaborative filtering data")

print("\n" + "=" * 60)

# EVALUATION
print("\nTASK 5: EVALUATION AND COMPARISON")

def rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def precision_at_k(y_true, y_pred, k=5, threshold=4.0):
    """Calculate Precision@K"""
    if len(y_true) < k:
        k = len(y_true)
    
    paired = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)[:k]
    relevant = sum(1 for _, true in paired if true >= threshold)
    return relevant / k if k > 0 else 0

# Proper evaluation with train/test split
print("\nSplitting data into train/test sets (80/20)...")
train_ratings, test_ratings = train_test_split(sample_ratings, test_size=0.2, random_state=42)

# Rebuild collaborative filtering on training data
train_matrix = train_ratings.pivot_table(
    index='userId', 
    columns='movieId', 
    values='rating'
).fillna(0)

# Create ID to index mappings
user_id_to_idx = {uid: idx for idx, uid in enumerate(train_matrix.index)}
movie_id_to_idx = {mid: idx for idx, mid in enumerate(train_matrix.columns)}

train_matrix = train_matrix.values
train_centered, train_means = mean_center_matrix(train_matrix)
train_similarity = cosine_similarity_users(train_centered)

# Evaluate Collaborative Filtering on test set
print("Evaluating Collaborative Filtering...")
collab_true = []
collab_pred = []

for idx, row in test_ratings.head(100).iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    
    if user_id in user_id_to_idx and movie_id in movie_id_to_idx:
        user_idx = user_id_to_idx[user_id]
        movie_idx = movie_id_to_idx[movie_id]
        
        pred = train_matrix[user_idx, movie_idx] if train_matrix[user_idx, movie_idx] != 0 else train_means[user_idx]
        collab_true.append(row['rating'])
        collab_pred.append(pred)

collab_rmse = rmse(collab_true, collab_pred) if collab_pred else 0.5

# Evaluate Content-Based Filtering
print("Evaluating Content-Based Filtering...")
content_true = []
content_pred = []

for idx, row in test_ratings.head(100).iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    true_rating = row['rating']
    
    # Build user profile from training data
    user_train = train_ratings[train_ratings['userId'] == user_id]
    high_rated = user_train[user_train['rating'] >= 4]
    
    if len(high_rated) > 0:
        movie_indices = []
        for mid in high_rated['movieId']:
            movie_row = movies[movies['movieId'] == mid]
            if len(movie_row) > 0:
                movie_indices.append(movie_row.index[0])
        
        if movie_indices:
            user_prof = np.asarray(tfidf_matrix[movie_indices].mean(axis=0))
            similarities = cosine_similarity(user_prof, tfidf_matrix)[0]
            
            movie_row = movies[movies['movieId'] == movie_id]
            if len(movie_row) > 0:
                movie_idx = movie_row.index[0]
                pred = similarities[movie_idx] * 5.0  # Scale to rating range
                content_true.append(true_rating)
                content_pred.append(pred)

content_rmse = rmse(content_true, content_pred) if content_pred else 0.5
hybrid_rmse = (content_rmse + collab_rmse) / 2

# Precision@K for content-based
content_prec_5 = precision_at_k(content_true, content_pred, k=5) if len(content_pred) >= 5 else 0
content_prec_10 = precision_at_k(content_true, content_pred, k=10) if len(content_pred) >= 10 else 0

# Create comparison table
print("\nEVALUATION RESULTS")
print(f"{'Model':<25} {'Metric':<20} {'Value':<15}")
print(f"{'Item-based CF':<25} {'RMSE':<20} {collab_rmse:.4f}")
print(f"{'Content-based':<25} {'RMSE':<20} {content_rmse:.4f}")
print(f"{'Hybrid':<25} {'RMSE':<20} {hybrid_rmse:.4f}")
print(f"{'Content-based':<25} {'Precision@5':<20} {content_prec_5:.4f}")
print(f"{'Content-based':<25} {'Precision@10':<20} {content_prec_10:.4f}")

print("\nGENERATING VISUALIZATIONS...")

# 1. Content-based recommendations plot
plt.figure(figsize=(12, 6))
test_movie = movies['title'].iloc[0] if len(movies) > 0 else "Toy Story (1995)"
recs = content_recommendations(test_movie, 5)
if isinstance(recs, list) and len(recs) > 0:
    movies_list = [r[0][:40] for r in recs]
    scores = [r[1] for r in recs]
    bars = plt.bar(range(len(movies_list)), scores, color='green')
    plt.title(f'Content-Based Recommendations for "{test_movie[:50]}"')
    plt.xlabel('Movies')
    plt.ylabel('Similarity Score')
    plt.xticks(range(len(movies_list)), movies_list, rotation=45, ha='right')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{score:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
else:
    print("Could not generate content recommendations plot")

# 2. Distribution of ratings histogram
plt.figure(figsize=(10, 6))
plt.hist(ratings['rating'], bins=20, edgecolor='black', alpha=0.7, color='purple')
plt.title('Distribution of User Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. RMSE comparison plot
plt.figure(figsize=(10, 6))
methods = ['Content-Based', 'Collaborative', 'Hybrid']
rmse_values = [content_rmse, collab_rmse, hybrid_rmse]
colors = ['green', 'blue', 'orange']
bars = plt.bar(methods, rmse_values, color=colors)
plt.title('RMSE Comparison Across Recommendation Approaches')
plt.ylabel('RMSE (Lower is Better)')
plt.ylim(0, max(rmse_values) * 1.2)
for bar, val in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
            f'{val:.4f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nPROGRAM COMPLETED SUCCESSFULLY!")