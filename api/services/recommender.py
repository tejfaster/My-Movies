import joblib
import pandas as pd
import numpy as np
from pathlib import Path


#getting project root directory
base_dir = Path(__file__).resolve().parent.parent.parent 

models_dir = base_dir/"models"
data_dir = base_dir/"data"/"ml-latest"

print("...Loading models...")

als_model = joblib.load(models_dir/"als_model.pkl")
xgb_model = joblib.load(models_dir/"xgboost_ranker.pkl")

user_id_map = joblib.load(models_dir/"user_id_map.pkl")
movie_id_map = joblib.load(models_dir/"movie_id_map.pkl")
sparse_matrix = joblib.load(models_dir/"sparse_matrix.pkl")

movie_idx_to_id = {v: k for k, v in movie_id_map.items()}

movies = pd.read_csv(data_dir/"movies.csv")
movie_id_to_title = dict(zip(movies['movieId'],movies['title']))

user_embeddings = als_model.user_factors
movie_embeddings = als_model.item_factors

print("Models loaded successfully.")

def cosine_similarity(u,v):
    return np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))


# def recommend_movies(user_id,n=10):

#     if user_id not in user_id_map:
#         return []
    
#     user_idx = user_id_map[user_id]

#     movie_indices, als_scores = als_model.recommend(
#         user_idx,
#         sparse_matrix[user_idx],
#         N=100
#     )

#     if len(movie_indices) == 0:
#         return[]
    
#     features = []

#     user_vec = user_embeddings[user_idx]

#     for movie_idx, als_score in zip(movie_indices,als_scores):

#         movie_vec = movie_embeddings[movie_idx]

#         similarity = cosine_similarity(user_vec,movie_vec)

#         features.append({
#             "movie_idx":movie_idx,
#             "als_score":als_score,
#             "embedding_similarity":similarity
#         })

#     feature_df =pd.DataFrame(
#          features
#         #  columns=["movie_idx","als_score","embedding_similarity"]
#     )   
    
#     X = feature_df[["als_score","embedding_similarity"]]

#     scores = xgb_model.predict_proba(X)[:,1]

#     feature_df = feature_df.sort_values("xgb_score",ascending=False)

#     recommendations = []

#     for row in feature_df.head(n).itertuples():
#         movie_id = movie_idx_to_id[row.movie_idx]
#         title = movie_id_to_title.get(movie_id,"Unknown")

#         recommendations.append({
#             "movie":title,
#             "score":float(row.xgb_score)
#         })

#     return recommendations    

def recommend_movies(user_id, n=10):

    # Check user exists
    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]

    # Generate candidates using ALS
    movie_indices, als_scores = als_model.recommend(
        user_idx,
        sparse_matrix[user_idx],
        N=100
    )

    if len(movie_indices) == 0:
        return []

    # Prepare feature list
    rows = []

    user_vec = user_embeddings[user_idx]

    for movie_idx, als_score in zip(movie_indices, als_scores):

        movie_vec = movie_embeddings[movie_idx]

        similarity = cosine_similarity(user_vec, movie_vec)

        rows.append({
            "movie_idx": movie_idx,
            "als_score": float(als_score),
            "embedding_similarity": float(similarity)
        })

    # Create DataFrame
    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        return []

    # Prepare XGBoost input
    X = feature_df[["als_score", "embedding_similarity"]]

    # Predict ranking score
    xgb_scores = xgb_model.predict_proba(X)[:, 1]

    # Assign score column
    feature_df["xgb_score"] = xgb_scores

    # Sort safely
    feature_df = feature_df.sort_values(by="xgb_score", ascending=False)

    # Build response
    recommendations = []

    for row in feature_df.head(n).itertuples():

        movie_id = movie_idx_to_id[row.movie_idx]

        title = movie_id_to_title.get(movie_id, "Unknown")

        recommendations.append({
            "movie": title,
            "score": float(row.xgb_score)
        })

    return recommendations
