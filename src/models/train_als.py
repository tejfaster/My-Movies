import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import joblib 

print("...Loading train data....")

train = pd.read_csv("../../data/train.csv")

print("Train data shape:",train.shape)

print("Creating ID mappings...")

user_ids = train["userId"].unique()
movie_ids = train["movieId"].unique()

user_id_map = {id: i for i, id in enumerate(user_ids)}
movie_id_map = {id: i for i, id in enumerate(movie_ids)}

train['user_idx'] = train['userId'].map(user_id_map)
train['movie_idx'] = train['movieId'].map(movie_id_map)

print("Creating sparse matrix...")

sparse_matrix = csr_matrix(
    (
        train['rating'],
        (train['user_idx'],train['movie_idx'])
    )
)

print("Sparse. matrix shape:",sparse_matrix.shape)

print("Training ALS model....")

model = implicit.als.AlternatingLeastSquares(
    factors=50,
    iterations =20,
    regularization =0.01
)

model.fit(sparse_matrix)

print("ALS training completed.")

print("Saving model...")

joblib.dump(model,"../../models/als_model.pkl")
joblib.dump(user_id_map,"../../models/user_id_map.pkl")
joblib.dump(movie_id_map,"../../models/movie_id_map.pkl")
joblib.dump(sparse_matrix,"../../models/sparse_matrix.pkl")

print("Model saved successfully.")