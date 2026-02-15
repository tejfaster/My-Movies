import joblib 
import pandas as pd

print("...Loading ALS model...")

model = joblib.load("../../models/als_model.pkl")
user_id_map = joblib.load("../../models/user_id_map.pkl")
movie_id_map = joblib.load("../../models/movie_id_map.pkl")
sparse_matrix = joblib.load("../../models/sparse_matrix.pkl")

# reverse mapping
movie_idx_to_id = {v: k for k,v in movie_id_map.items()}

print("Model loaded successfully.")

movies = pd.read_csv("../../data/ml-latest/movies.csv")

movie_id_to_title = dict(zip(movies['movieId'],movies['title']))

def recommend_movies(user_id,n=10):
    if user_id not in user_id_map:
        print("User not found")
        return
    
    user_idx = user_id_map[user_id]

    recommendations = model.recommend(
        user_idx,
        sparse_matrix[user_idx],
        N=n
        )

    recommended_movies = []
    
    movie_indices, scores = recommendations

    for movie_idx, score in zip(movie_indices, scores):
        movie_id = movie_idx_to_id[movie_idx]
        title = movie_id_to_title.get(movie_id,"Unknown")
        recommended_movies.append((title,score))

    return recommended_movies

user_id = list(user_id_map.keys())[0]

recommendations = recommend_movies(user_id,10)

print("\n Recommedations for user:",user_id)

for movie,score in recommendations:
    print(f"{movie} | score: {score:.4f}")