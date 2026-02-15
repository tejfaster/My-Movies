import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

print("...Loading ranking feature...")

features = pd.read_csv("../../data/ranking_features.csv")

print("Feature shape:",features.shape)
print(features.head())

X = features[['als_score',"embedding_similarity"]]
y = features["label"]

print("Positive samples:",y.sum())
print("Negative samples:",len(y) - y.sum())

print("Training XGBoost ranker...")

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators = 100,
    max_depth = 6,
    learning_rate = 0.1,
    subsample=0.8,
    colsample_bytree =0.8,
    tree_method="hist"
)

model.fit(X,y)

print("Traiing Completed.")

joblib.dump(model,"../../models/xgboost_ranker.pkl")

print("Ranking model saved.")