import sqlite3
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "subscription_analytics.db")
SQL_DIR = SCRIPT_DIR
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

queries = {
    "mrr_trend.csv": "01_mrr_trend.sql",
    "monthly_churn_rate.csv": "02_monthly_churn_rate.sql",
    "cohort_retention.csv": "03_cohort_retention.sql",
    "churn_by_plan.csv": "04_churn_by_plan.sql",
}

for out_name, sql_file in queries.items():
    with open(os.path.join(SQL_DIR, sql_file)) as f:
        query = f.read()
    df = pd.read_sql_query(query, conn)
    out_path = os.path.join(OUT_DIR, out_name)
    df.to_csv(out_path, index=False)
    print(f"{out_name}: {len(df)} rows -> {out_path}")

conn.close()
