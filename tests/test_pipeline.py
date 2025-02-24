"""tests/test_pipeline.py — Unit tests. Run: python -m pytest tests/ -v"""
import sys, pytest, numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.feature_engineering import build_genre_matrix, compute_tfidf_sentiment
from src.bias_correction     import correct_popularity_bias
from src.cold_start          import cold_start_recommendations
from src.evaluate            import compute_rmse_mae, ndcg_at_k


def make_movies():
    return pd.DataFrame({
        "movieId": [1, 2, 3],
        "title":   ["Action Hero (1990)", "Love Story (1995)", "Dark Comedy (2000)"],
        "genres":  ["Action|Adventure", "Romance|Drama", "Comedy"]
    })

def make_ratings():
    return pd.DataFrame({
        "movieId": [1, 1, 2, 3],
        "userId":  [1, 2, 1, 2],
        "rating":  [5, 4, 3, 2]
    })

class TestGenreMatrix:
    def test_builds_correctly(self):
        gm, genres = build_genre_matrix(make_movies())
        assert 1 in gm and 2 in gm and 3 in gm
        assert len(genres) > 0
        assert all(len(v) == len(genres) for v in gm.values())

    def test_binary_values(self):
        gm, _ = build_genre_matrix(make_movies())
        for v in gm.values():
            assert set(v).issubset({0.0, 1.0})

class TestTfIdf:
    def test_adds_sentiment_column(self):
        movies = make_movies()
        result, _ = compute_tfidf_sentiment(movies, max_features=10)
        assert "sentiment_score" in result.columns

    def test_scores_are_numeric(self):
        movies = make_movies()
        result, _ = compute_tfidf_sentiment(movies, max_features=10)
        assert result["sentiment_score"].dtype in [float, "float64"]

class TestBiasCorrection:
    def test_returns_sorted_by_diversity(self):
        recs    = [(1, 0.9), (2, 0.8), (3, 0.7)]
        ratings = make_ratings()
        result  = correct_popularity_bias(recs, ratings, alpha=0.5)
        scores  = [r["diversity_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_movies_present(self):
        recs    = [(1, 0.9), (2, 0.8)]
        ratings = make_ratings()
        result  = correct_popularity_bias(recs, ratings, alpha=0.3)
        ids     = [r["movie_id"] for r in result]
        assert 1 in ids and 2 in ids

class TestMetrics:
    def test_rmse_zero_for_perfect(self):
        y = [0.5, 0.7, 0.3]
        rmse, mae = compute_rmse_mae(y, y)
        assert rmse == 0.0 and mae == 0.0

    def test_ndcg_perfect(self):
        score = ndcg_at_k([1, 1, 1, 0, 0], k=5)
        assert score == pytest.approx(1.0)

    def test_ndcg_zero(self):
        score = ndcg_at_k([0, 0, 0], k=3)
        assert score == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
