"""
Builds chart images from the query outputs in outputs/*.csv.
Saves PNGs to outputs/charts/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# 1. MRR trend
df = pd.read_csv(os.path.join(OUT_DIR, "mrr_trend.csv"))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["billing_month"], df["total_mrr"], linewidth=2, color="#27ae60")
ax.fill_between(range(len(df)), df["total_mrr"], alpha=0.15, color="#27ae60")
ax.set_xlabel("Month")
ax.set_ylabel("Monthly Recurring Revenue (EUR)")
ax.set_title("MRR Trend")
step = max(1, len(df) // 12)
ax.set_xticks(range(0, len(df), step))
ax.set_xticklabels(df["billing_month"][::step], rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "mrr_trend.png"), dpi=150)
plt.close()

# 2. Monthly churn rate
df = pd.read_csv(os.path.join(OUT_DIR, "monthly_churn_rate.csv"))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["billing_month"], df["churn_rate_pct"], linewidth=2, color="#c0392b", marker="o", markersize=3)
ax.set_xlabel("Month")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Monthly Churn Rate")
step = max(1, len(df) // 12)
ax.set_xticks(range(0, len(df), step))
ax.set_xticklabels(df["billing_month"][::step], rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "monthly_churn_rate.png"), dpi=150)
plt.close()

# 3. Churn by plan
df = pd.read_csv(os.path.join(OUT_DIR, "churn_by_plan.csv"))
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(df["plan"], df["churn_rate_pct"], color=["#c0392b", "#e67e22", "#27ae60"])
ax.set_xlabel("Plan")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Plan Tier")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "churn_by_plan.png"), dpi=150)
plt.close()

# 4. Cohort retention heatmap (pivot: cohort_month x months_since_signup)
df = pd.read_csv(os.path.join(OUT_DIR, "cohort_retention.csv"))
pivot = df.pivot(index="cohort_month", columns="months_since_signup", values="retention_pct")
pivot = pivot.iloc[:, :13]  # first 12 months since signup

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(0, len(pivot.index), max(1, len(pivot.index) // 20)))
ax.set_yticklabels(pivot.index[::max(1, len(pivot.index) // 20)])
ax.set_xlabel("Months Since Signup")
ax.set_ylabel("Signup Cohort")
ax.set_title("Cohort Retention (%)")
fig.colorbar(im, ax=ax, label="Retention %")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "cohort_retention_heatmap.png"), dpi=150)
plt.close()

print(f"4 charts saved to {CHART_DIR}")
