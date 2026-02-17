import pandas as pd
import joblib
import os


def load_model_and_data():

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "actor_genre_investment_model.pkl"
    )

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "actor_genre_features.csv"
    )

    print("Loading model...")
    model = joblib.load(model_path)

    print("Loading actor-genre dataset...")
    df = pd.read_csv(data_path)

    return model, df


def predict_actor_genre_score(actor_name, genre):

    model, df = load_model_and_data()

    # find actor + genre row
    row = df[
        (df["actor_name"].str.lower() == actor_name.lower())
        &
        (df["genre"].str.lower() == genre.lower())
    ]

    if row.empty:
        print(f"\nNo data found for Actor='{actor_name}', Genre='{genre}'")
        return

    feature_columns = [

        "actor_genre_watch_count",
        "actor_genre_votes",
        "actor_genre_avg_rating",
        "career_length",
        "recency_score",
        "actor_genre_movie_count"

    ]

    X = row[feature_columns]

    prediction = model.predict(X)[0]

    print("\n========== INVESTMENT PREDICTION ==========")
    print(f"Actor: {actor_name}")
    print(f"Genre: {genre}")
    print(f"Predicted Investment Score: {prediction:.4f}")

    return prediction


def recommend_best_actors_by_genre(genre, top_n=10):

    model, df = load_model_and_data()

    genre_df = df[df["genre"].str.lower() == genre.lower()].copy()

    if genre_df.empty:
        print(f"No actors found for genre: {genre}")
        return

    feature_columns = [

        "actor_genre_watch_count",
        "actor_genre_votes",
        "actor_genre_avg_rating",
        "career_length",
        "recency_score",
        "actor_genre_movie_count"

    ]

    genre_df["predicted_score"] = model.predict(
        genre_df[feature_columns]
    )

    genre_df = genre_df.sort_values(
        "predicted_score",
        ascending=False
    )

    print(f"\n========== TOP {top_n} ACTORS FOR {genre.upper()} ==========\n")

    for i, row in enumerate(
        genre_df.head(top_n).itertuples(),
        1
    ):
        print(f"{i}. {row.actor_name} — Score: {row.predicted_score:.4f}")


if __name__ == "__main__":

    # Example predictions

    predict_actor_genre_score("Tom Cruise", "Action")

    recommend_best_actors_by_genre("Action", top_n=10)
