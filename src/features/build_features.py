import pandas as pd
import numpy as np 
import joblib

print("...Loading data...")

train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("../../data/test.csv")

model = joblib.load("../../models/als_model.pkl")
user_id_map = joblib.load("../../models/user_id_map.pkl")
movie_id_map = joblib.load("../../models/movie_id_map.pkl")
sparse_matrix = joblib.load("../../models/sparse_matrix.pkl")

movie_idx_to_id = {v: k for k, v in movie_id_map.items()}
movie_id_to_idx = movie_id_map

user_embeddings = model.user_factors
movie_embeddings = model.item_factors

print("User embedding shape:",user_embeddings.shape)
print("Movie embedding shape:",movie_embeddings.shape)

print("Preparing training sample...")

test_user_movies = test.groupby("userId")["movieId"].apply(set).to_dict()

samples = []

# users = list(user_id_map.keys())[:5000]
users = list(test_user_movies.keys())[:5000]

for user_id in users:

    if user_id not in user_id_map:
        continue

    user_idx = user_id_map[user_id]

    # Positive sample (movies actually watched in future)
    positive_movies = test_user_movies[user_id]

    for movie_id in positive_movies:
        
        if movie_id not in movie_id_map:
            continue
        
        movie_idx = movie_id_map[movie_id]

        user_vec = user_embeddings[user_idx]
        movie_vec = movie_embeddings[movie_idx]

        als_score = np.dot(user_vec,movie_vec)

        samples.append([
            user_idx,
            movie_idx,
            als_score,
            1
        ])

    # Negative sample (AlS recommendation not watched)
    movie_indices,scores = model.recommend(
        user_idx,
        sparse_matrix[user_idx],
        N=50
    )

    for movie_idx, score in zip(movie_indices, scores):

        movie_id = movie_idx_to_id[movie_idx]

        # label = 1 if movie_id in test_user_movies.get(user_id,set()) else 0
        if movie_id in positive_movies:
            continue

        samples.append([
            user_idx,
            movie_idx,
            score,
            0
        ])

features = pd.DataFrame(
    samples,
    columns=["user_idx","movie_idx","als_score","label"]
)        

print(features.head())

def cosine_similarity(u,v):
    return np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))

similarities = []

for row in features.itertuples():

    user_vec = user_embeddings[row.user_idx]
    movie_vec = movie_embeddings[row.movie_idx]

    sim = cosine_similarity(user_vec,movie_vec)

    similarities.append(sim)

features["embedding_similarity"] = similarities

features.to_csv("../../data/ranking_features.csv",index=False)

print("Features saved.")