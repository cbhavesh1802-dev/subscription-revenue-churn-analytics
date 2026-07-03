import sqlite3
import pandas as pd
import os

DB_PATH = "/home/claude/projects/02-subscription-churn-analytics/data/subscription_analytics.db"
SQL_DIR = "/home/claude/projects/02-subscription-churn-analytics/sql"
OUT_DIR = "/home/claude/projects/02-subscription-churn-analytics/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

queries = {
    "mrr_trend.csv": "01_mrr_trend.sql",
    "monthly_churn_rate.csv": "02_monthly_churn_rate.sql",
    "cohort_retention.csv": "03_cohort_retention.sql",
    "churn_by_plan.csv": "04_churn_by_plan.sql",
}

for out_name, sql_file in queries.items():
    with open(f"{SQL_DIR}/{sql_file}") as f:
        query = f.read()
    df = pd.read_sql_query(query, conn)
    out_path = f"{OUT_DIR}/{out_name}"
    df.to_csv(out_path, index=False)
    print(f"{out_name}: {len(df)} rows -> {out_path}")

conn.close()
