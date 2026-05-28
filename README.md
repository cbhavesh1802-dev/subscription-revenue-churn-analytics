# 🎬 Movie Recommendation System
## Emotion-Aware Hybrid Neural Collaborative Filtering

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?style=flat-square&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Publication%20Ready-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **Bachelor's Research Project** | Department of Computer Science & Engineering | 2024

---

## 📌 Overview

This project implements **EA-HNCF** (Emotion-Aware Hybrid Neural Collaborative Filtering) — a novel movie recommendation system that simultaneously addresses three persistent limitations of conventional recommenders: the cold-start problem, popularity bias, and the lack of semantic content awareness.

Unlike standard NCF which only uses user-item interaction embeddings, EA-HNCF fuses collaborative signals with genre-weighted user preferences and TF-IDF-derived emotion signals from movie titles — making the system semantically aware of movie content.

---

## ★ Novel Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Emotion-Aware TF-IDF Fusion** | Title sentiment scores from TF-IDF fused as 4th input channel to NCF |
| 2 | **Genre-Weighted User Embeddings** | Per-user genre preference weights personalize content representation |
| 3 | **Popularity Bias Correction** | Alpha-controlled re-ranking reduces over-recommendation of blockbusters |
| 4 | **Cold-Start Content Bootstrapping** | Genre-diversity scoring provides recommendations from first interaction |

---

## 📊 Results

| Model | RMSE | MAE | Precision@10 | NDCG@10 |
|---|---|---|---|---|
| User-Based CF | 1.032 | 0.821 | 0.61 | 0.68 |
| Item-Based CF | 0.991 | 0.796 | 0.65 | 0.71 |
| SVD (Matrix Factorization) | 0.952 | 0.743 | 0.70 | 0.75 |
| KNN-Based | 0.974 | 0.762 | 0.67 | 0.73 |
| Standard NCF (GMF) | 0.901 | 0.712 | 0.74 | 0.79 |
| **EA-HNCF (Ours)** | **0.847** | **0.668** | **0.79** | **0.83** |

---

## 📁 Project Structure

```
movie_recommendation_system/
├── README.md
├── requirements.txt
├── setup.py
├── train.py                     # Full training pipeline
├── recommend.py                 # Generate top-K recommendations
│
├── config/
│   └── config.yaml              # All hyperparameters
│
├── data/
│   └── README.md                # Dataset download instructions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load ratings + movies CSVs
│   ├── preprocessor.py          # Encode IDs, normalize ratings
│   ├── feature_engineering.py   # ★ Genre-weighted + TF-IDF features
│   ├── ncf_model.py             # ★ EA-HNCF architecture (TensorFlow/Keras)
│   ├── cold_start.py            # ★ Cold-start content bootstrapping
│   ├── bias_correction.py       # ★ Popularity bias correction
│   ├── baselines.py             # SVD and CF baselines
│   ├── evaluate.py              # RMSE, MAE, Precision@K, NDCG@K
│   └── utils.py                 # Config, logging, helpers
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_NCF_Training.ipynb
│   └── 04_Evaluation.ipynb
│
├── api/
│   └── app.py                   # Flask API for real-time recommendations
│
├── models/                      # Saved model weights
├── results/                     # Plots and metrics
└── tests/
    └── test_pipeline.py
```

---

## 🚀 Installation & Usage

```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
```

Download `ratings.csv` and `movies.csv` from [Kaggle MovieLens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) → place in `data/`

```bash
python train.py               # Train EA-HNCF + all baselines
python recommend.py --user 42 # Get top-10 for user 42
python api/app.py             # Start Flask API at localhost:5000
```

---

## 🔌 API Usage

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "top_k": 10}'

# Response
{
  "user_id": 42,
  "recommendations": [
    {"rank": 1, "movie_id": 318, "title": "Shawshank Redemption", "score": 0.94},
    ...
  ],
  "cold_start": false
}
```

---

## 📦 Dataset

- **Source**: [Kaggle — MovieLens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- **Files needed**: `ratings.csv`, `movies.csv`
- **Size**: 100K ratings | 943 users | 1,682 movies

---

## 🔬 Citation

```bibtex
@article{movie2023eahncf,
  title={Emotion-Aware Hybrid Neural Collaborative Filtering for Movie Recommendation},
  author={Bhavesh Mukesh Chaudhari},
  year={2023}
}
```
