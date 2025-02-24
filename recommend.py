"""
recommend.py — Generate top-K recommendations for a user.
Usage: python recommend.py --user 42 --top_k 10
"""
import sys, argparse
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.preprocessor       import MoviePreprocessor
from src.ncf_model          import load_model
from src.feature_engineering import build_genre_matrix, compute_tfidf_sentiment, prepare_inputs
from src.bias_correction    import correct_popularity_bias
from src.cold_start         import cold_start_recommendations
from src.data_loader        import load_dataset
from src.utils              import load_config


def recommend(user_id, top_k=10, alpha=0.3):
    cfg     = load_config()
    ratings, movies = load_dataset(cfg["data"]["ratings_path"], cfg["data"]["movies_path"])
    prep    = MoviePreprocessor().load()
    model   = load_model()
    genre_matrix, all_genres = build_genre_matrix(movies)
    movies, _ = compute_tfidf_sentiment(movies)
    n_genres  = len(all_genres)

    # Check if user is known
    if user_id not in prep.user_enc.classes_:
        print(f"  New user {user_id} — using Cold-Start recommendations (★ Novel)")
        recs = cold_start_recommendations(genre_matrix, movies, top_k)
        for i, r in enumerate(recs, 1):
            print(f"  {i:>2}. {r['title']}")
        return recs

    user_idx = prep.user_enc.transform([user_id])[0]
    all_movie_ids  = movies["movieId"].values
    all_movie_idxs = prep.movie_enc.transform(
        [m for m in all_movie_ids if m in prep.movie_enc.classes_])

    # Build input for all movies for this user
    temp_df = pd.DataFrame({
        "userId": user_id, "movieId": all_movie_ids[:len(all_movie_idxs)],
        "user_idx": user_idx, "movie_idx": all_movie_idxs,
        "rating_norm": 0.0
    })
    X, _ = prepare_inputs(temp_df, genre_matrix, movies, n_genres)
    scores = model.predict(X, verbose=0).flatten()

    # Popularity bias correction (★ Novel)
    raw_recs   = list(zip(all_movie_ids[:len(scores)], scores))
    corrected  = correct_popularity_bias(raw_recs, ratings, alpha=alpha)[:top_k]

    # Add titles
    mid_to_title = dict(zip(movies["movieId"], movies["title"]))
    print(f"\n  Top-{top_k} Recommendations for User {user_id}:")
    for i, r in enumerate(corrected, 1):
        title = mid_to_title.get(r["movie_id"], "Unknown")
        print(f"  {i:>2}. {title[:50]:<50}  score={r['ncf_score']:.3f}")
    return corrected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",  type=int, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()
    recommend(args.user, args.top_k, args.alpha)
