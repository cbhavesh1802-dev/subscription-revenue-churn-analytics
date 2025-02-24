"""
preprocessor.py
===============
Encodes user/movie IDs and normalizes ratings.

WHY LabelEncoder?
  Embedding layers require contiguous integer indices starting at 0.
  User IDs may not be contiguous (e.g., 1, 5, 23...). LabelEncoder
  maps them to 0, 1, 2, ... for the embedding lookup.

WHY MinMaxScaler?
  The NCF output layer uses sigmoid → values in [0,1].
  Training is more stable when labels match output range.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib, os


class MoviePreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.user_enc  = LabelEncoder()
        self.movie_enc = LabelEncoder()
        self.scaler    = MinMaxScaler()
        self.test_size = test_size
        self.rs        = random_state

    def fit_transform(self, ratings):
        ratings = ratings.copy()
        ratings["user_idx"]   = self.user_enc.fit_transform(ratings["userId"])
        ratings["movie_idx"]  = self.movie_enc.fit_transform(ratings["movieId"])
        ratings["rating_norm"]= self.scaler.fit_transform(ratings[["rating"]])

        self.n_users  = ratings["user_idx"].max() + 1
        self.n_movies = ratings["movie_idx"].max() + 1

        # Stratified split — each user in both train and test
        train, test = train_test_split(
            ratings, test_size=self.test_size,
            random_state=self.rs,
            stratify=ratings["user_idx"] % 5
        )
        print(f"  Train: {len(train):,} | Test: {len(test):,}")
        print(f"  Users: {self.n_users} | Movies: {self.n_movies}")
        return train, test

    def save(self, directory="models/"):
        os.makedirs(directory, exist_ok=True)
        joblib.dump({
            "user_enc": self.user_enc,
            "movie_enc": self.movie_enc,
            "scaler": self.scaler,
            "n_users": self.n_users,
            "n_movies": self.n_movies
        }, f"{directory}/preprocessor.pkl")

    def load(self, directory="models/"):
        d = joblib.load(f"{directory}/preprocessor.pkl")
        self.user_enc  = d["user_enc"]
        self.movie_enc = d["movie_enc"]
        self.scaler    = d["scaler"]
        self.n_users   = d["n_users"]
        self.n_movies  = d["n_movies"]
        return self
