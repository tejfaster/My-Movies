import pandas as pd
import joblib 
import numpy as np

print("...Loading data...")

train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("../../data/test.csv")

model = joblib.load("../../models/als_model.pkl")
user_id_map = joblib.load("../../models/user_id_map.pkl")
movie_id_map = joblib.load("../../models/movie_id_map.pkl")
sparese_matrix = joblib.load("../../models/sparse_matrix.pkl")

movie_idx_to_id = {v: k for k, v in movie_id_map.items()}

print("Preparing test interactions...")

test_user_movies = test.groupby('userId')['movieId'].apply(set).to_dict()

def precision_at_k(user_id,k =10):

    if user_id not in user_id_map:
        return None
    
    user_idx = user_id_map[user_id]

    movie_indices, scores = model.recommend(
        user_idx,
        sparese_matrix[user_idx],
        N=k
    )

    recommended_movie_ids = {
        movie_idx_to_id[idx]
        for idx in movie_indices
    }

    actual_movie_ids = test_user_movies.get(user_id,set())

    if len(actual_movie_ids) == 0:
        return None
    
    hits = len(recommended_movie_ids & actual_movie_ids)

    return hits / k

print("Evaluating model...")

users = list(test_user_movies.keys())[:1000]

precisions = []

for user_id in users:

    p = precision_at_k(user_id, k = 10)

    if p is not None:
        precisions.append(p)

mean_precision = np.mean(precisions)

print(f"\nPrecision@10:{mean_precision:.4f}")