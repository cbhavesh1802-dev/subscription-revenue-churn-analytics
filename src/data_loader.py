"""
data_loader.py — Load and validate MovieLens dataset.
"""
import pandas as pd, os

def load_dataset(ratings_path, movies_path):
    for p in [ratings_path, movies_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File not found: {p}\n"
                "Download from: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset"
            )
    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)
    print(f"  Ratings: {ratings.shape} | Movies: {movies.shape}")
    print(f"  Users: {ratings['userId'].nunique():,} | Items: {ratings['movieId'].nunique():,}")
    print(f"  Ratings: {ratings['rating'].min()}–{ratings['rating'].max()} | Avg: {ratings['rating'].mean():.2f}")
    return ratings, movies
