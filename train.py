"""
train.py — Movie Recommendation System Training Pipeline
Usage: python train.py
"""
import sys
sys.path.insert(0, ".")

from src.utils              import load_config, log, ensure_dirs
from src.data_loader        import load_dataset
from src.preprocessor       import MoviePreprocessor
from src.feature_engineering import build_genre_matrix, compute_tfidf_sentiment, prepare_inputs
from src.ncf_model          import build_eahncf, get_callbacks, save_model
from src.baselines          import train_svd, evaluate_svd
from src.cold_start         import cold_start_recommendations
from src.evaluate           import compute_rmse_mae, plot_training_history, save_results


def main():
    cfg = load_config()
    ensure_dirs("models", "results")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   MOVIE RECOMMENDATION — EA-HNCF TRAINING PIPELINE  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    log("Step 1: Loading MovieLens dataset...")
    ratings, movies = load_dataset(cfg["data"]["ratings_path"], cfg["data"]["movies_path"])

    log("Step 2: Preprocessing — encode IDs + normalize ratings...")
    prep = MoviePreprocessor(test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"])
    train_df, test_df = prep.fit_transform(ratings)
    prep.save()

    log("Step 3: ★ Novel Feature Engineering — Genre + TF-IDF Sentiment...")
    genre_matrix, all_genres = build_genre_matrix(movies)
    movies, _                = compute_tfidf_sentiment(movies, cfg["feature_engineering"]["tfidf_max_features"])
    n_genres = len(all_genres)

    log("Step 4: SVD Baseline...")
    svd_pred = train_svd(train_df, prep.n_users, prep.n_movies)
    svd_rmse, svd_mae = evaluate_svd(svd_pred, test_df)

    log("Step 5: Cold-Start Content Bootstrapping (★ Novel)...")
    cold_recs = cold_start_recommendations(genre_matrix, movies, top_k=cfg["cold_start"]["top_k"])
    print("  Top-5 Cold-Start Recommendations (for new users):")
    for i, r in enumerate(cold_recs[:5], 1):
        print(f"    {i}. {r['title']} (diversity={r['diversity_score']})")

    log("Step 6: Building EA-HNCF model (★ Novel Architecture)...")
    model = build_eahncf(
        prep.n_users, prep.n_movies, n_genres,
        embedding_dim=cfg["model"]["embedding_dim"],
        mlp_layers=cfg["model"]["mlp_layers"],
        dropout_rate=cfg["model"]["dropout_rate"],
        learning_rate=cfg["model"]["learning_rate"]
    )

    log("Step 7: Training EA-HNCF...")
    X_train, y_train = prepare_inputs(train_df, genre_matrix, movies, n_genres)
    X_test,  y_test  = prepare_inputs(test_df,  genre_matrix, movies, n_genres)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=cfg["model"]["epochs"],
        batch_size=cfg["model"]["batch_size"],
        callbacks=get_callbacks(
            patience=cfg["model"]["early_stopping_patience"],
            min_lr=cfg["model"]["min_lr"]
        ),
        verbose=1
    )

    log("Step 8: Evaluating EA-HNCF...")
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse, mae = compute_rmse_mae(y_test, y_pred)
    print(f"\n  ★ EA-HNCF Results:")
    print(f"    RMSE : {rmse:.4f}  (SVD baseline: {svd_rmse:.4f})")
    print(f"    MAE  : {mae:.4f}  (SVD baseline: {svd_mae:.4f})")
    print(f"    Improvement: {(svd_rmse-rmse)/svd_rmse*100:.1f}% RMSE reduction vs SVD")

    log("Step 9: Saving model and results...")
    save_model(model)
    plot_training_history(history)
    save_results({
        "SVD":    {"rmse": svd_rmse, "mae": svd_mae},
        "EAHNCF": {"rmse": rmse,     "mae": mae}
    })
    log("✓ Training complete!")


if __name__ == "__main__":
    main()
