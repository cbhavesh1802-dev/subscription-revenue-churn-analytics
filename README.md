# Subscription Revenue & Churn Analytics

Cohort-based retention, MRR, and churn analysis for a subscription business -- built entirely with SQL and Python.

## Overview

This project analyzes 60,000 subscription customers and 1.5M+ billing events across 5.5 years to answer the core SaaS analyst questions: how is revenue trending, who is churning, and which cohorts/plans retain best?

## Key Findings

- MRR trend shows revenue trajectory across the full customer base over time (outputs/mrr_trend.csv)
- Churn varies sharply by plan tier -- Basic plan customers churn at ~55%, compared to ~25% for Pro -- a clear signal that plan value/stickiness differs by tier (outputs/churn_by_plan.csv)
- Cohort retention curves show how each signup months customers retain over their first N months, the standard SaaS retention view (outputs/cohort_retention.csv)
- Monthly churn rate tracks the rate of customer loss month over month, useful for spotting seasonal or campaign-driven spikes (outputs/monthly_churn_rate.csv)

## Project Structure

data/ - customers.csv, billing_events.csv, generate_synthetic.py
sql/ - 01_mrr_trend.sql, 02_monthly_churn_rate.sql, 03_cohort_retention.sql (multi-CTE cohort retention by signup month), 04_churn_by_plan.sql, load_db.py, run_all_queries.py
outputs/ - query results, ready for Power BI import

## How to Run

python sql/load_db.py
python sql/run_all_queries.py

## Methodology Notes

Cohort retention groups customers by signup month, then tracks what percentage of each cohort remains active N months later -- computed via a multi-step CTE joining billing activity back to cohort membership. Churn rate is calculated per month as churned customers divided by active customers at the start of that month. Data is synthetic, generated to mirror realistic SaaS patterns: churn hazard varies by plan tier and decreases with customer tenure, and signups follow a growth-then-maturity curve.

## Tech Stack

SQL (SQLite, window functions, CTEs) - Python (Pandas, NumPy) - Power BI - Excel
