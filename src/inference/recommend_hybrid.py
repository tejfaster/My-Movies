import joblib
import pandas as pd
import numpy as np

print("...Loading models...")

als_model = joblib.load("../../models/als_model.pkl")
xgb_model = joblib.load("../../models/xgboost_ranker.pkl")

user_id_map = joblib.load("../../models/user_id_map.pkl")
movie_id_map = joblib.load("../../models/movie_id_map.pkl")
sparese_matrix = joblib.load("../../models/sparse_matrix.pkl")

movie_idx_to_id = {v: k for k, v in movie_id_map.items()}
movie_id_to_idx = movie_id_map

print("Models loaded.")

movies = pd.read_csv("../../data/ml-latest/movies.csv")
movie_id_to_title = dict(zip(movies["movieId"],movies["title"]))

user_embeddings = als_model.user_factors
movie_embeddings = als_model.item_factors

def cosine_similarity(u,v):
    return np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))

def recommend_movies(user_id,n=10):

    if user_id not in user_id_map:
        print("User not found")
        return []
    
    user_idx = user_id_map[user_id]
    
    # Als candidqte genration
    movie_indices, als_scores = als_model.recommend(
        user_idx,
        sparese_matrix[user_idx],
        N=100
    )

    # Build feature dataframe
    features = []

    for movie_idx, als_score in zip(movie_indices, als_scores):

        user_vec = user_embeddings[user_idx]
        movie_vec = movie_embeddings[movie_idx]

        similarity = cosine_similarity(user_vec,movie_vec)

        features.append([
            movie_idx,
            als_score,
            similarity
        ])
    
    feature_df = pd.DataFrame(
        features,
        columns=["movie_idx","als_score","embedding_similarity"]
    )

    # XGBoost ranking
    X = feature_df[["als_score","embedding_similarity"]]

    scores = xgb_model.predict_proba(X)[:,1]

    feature_df["xgb_score"] = scores

    # sorting by ranking score
    feature_df = feature_df.sort_values("xgb_score",ascending=False) 

    # Convert to movie titles
    recommendations = []

    for row in feature_df.head(n).itertuples():

        movie_id = movie_idx_to_id[row.movie_idx]
        title = movie_id_to_title.get(movie_id,"Unknown")

        recommendations.append((title,row.xgb_score))

    return recommendations

user_id = list(user_id_map.keys())[0]

recommendations = recommend_movies(user_id,10)

print(f"\n Hybrid recommendation for user {user_id}:\n")

for movie,score in recommendations:
    print(f"{movie} | score: {score:.4f}")