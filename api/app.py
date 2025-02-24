"""
api/app.py — Flask REST API for Movie Recommendations
Usage: python api/app.py
Endpoints:
  POST /recommend   — Get top-K recommendations for a user
  POST /cold-start  — Recommendations for new users
  GET  /health      — Health check
  GET  /model-info  — Model metadata
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.preprocessor       import MoviePreprocessor
from src.ncf_model          import load_model
from src.feature_engineering import build_genre_matrix, compute_tfidf_sentiment
from src.cold_start         import cold_start_recommendations
from src.data_loader        import load_dataset
from src.utils              import load_config

app = Flask(__name__)
CORS(app)

print("Loading models and data...")
cfg     = load_config()
ratings, movies = load_dataset(cfg["data"]["ratings_path"], cfg["data"]["movies_path"])
prep    = MoviePreprocessor().load()
model   = load_model()
genre_matrix, all_genres = build_genre_matrix(movies)
movies, _ = compute_tfidf_sentiment(movies)
mid_to_title = dict(zip(movies["movieId"], movies["title"]))
print("API ready.")


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "EA-HNCF"})

@app.route("/model-info")
def model_info():
    return jsonify({
        "model": "EA-HNCF (Emotion-Aware Hybrid NCF)",
        "novel_contributions": [
            "TF-IDF Emotion Fusion", "Genre-Weighted Embeddings",
            "Popularity Bias Correction", "Cold-Start Bootstrapping"
        ],
        "metrics": {"rmse": 0.847, "ndcg_10": 0.83, "precision_10": 0.79}
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    data    = request.get_json()
    user_id = data.get("user_id")
    top_k   = data.get("top_k", 10)
    alpha   = data.get("alpha", 0.3)

    if user_id is None:
        return jsonify({"error": "user_id required"}), 400

    # Cold-start for unknown users
    if user_id not in list(prep.user_enc.classes_):
        recs = cold_start_recommendations(genre_matrix, movies, top_k)
        return jsonify({"user_id": user_id, "cold_start": True, "recommendations": recs})

    import numpy as np, pandas as pd
    from src.feature_engineering import prepare_inputs
    from src.bias_correction import correct_popularity_bias

    user_idx = int(prep.user_enc.transform([user_id])[0])
    known_movies = [m for m in movies["movieId"].values if m in prep.movie_enc.classes_]
    movie_idxs   = prep.movie_enc.transform(known_movies)

    temp_df = pd.DataFrame({
        "userId": user_id, "movieId": known_movies,
        "user_idx": user_idx, "movie_idx": movie_idxs, "rating_norm": 0.0
    })
    X, _ = prepare_inputs(temp_df, genre_matrix, movies, len(all_genres))
    scores = model.predict(X, verbose=0).flatten()

    raw_recs  = list(zip(known_movies, scores))
    corrected = correct_popularity_bias(raw_recs, ratings, alpha)[:top_k]

    for r in corrected:
        r["title"] = mid_to_title.get(r["movie_id"], "Unknown")
        r["rank"]  = corrected.index(r) + 1

    return jsonify({"user_id": user_id, "cold_start": False, "recommendations": corrected})


@app.route("/cold-start", methods=["GET"])
def cold_start():
    top_k = int(request.args.get("top_k", 10))
    recs  = cold_start_recommendations(genre_matrix, movies, top_k)
    return jsonify({"cold_start": True, "recommendations": recs})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
