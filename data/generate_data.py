"""
Generates sample subscription business data: customers, subscriptions,
and monthly billing events -- used to build and test cohort retention,
MRR, and churn analysis queries.
"""

import numpy as np
import pandas as pd
import os
from datetime import date, timedelta

np.random.seed(7)

N_CUSTOMERS = 60_000
START = date(2021, 1, 1)
MONTHS = 66

PLANS = {
    "Basic":   9.99,
    "Plus":    19.99,
    "Pro":     39.99,
}
PLAN_WEIGHTS = [0.5, 0.35, 0.15]

def month_add(d, n):
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, 1)

customers = []
subs = []
sub_id = 1

for cust_id in range(1, N_CUSTOMERS + 1):
    signup_month_idx = np.random.geometric(p=0.09)
    signup_month_idx = min(signup_month_idx, MONTHS - 2)
    signup_date = month_add(START, signup_month_idx)

    plan = np.random.choice(list(PLANS.keys()), p=PLAN_WEIGHTS)
    mrr = PLANS[plan]

    base_hazard = {"Basic": 0.055, "Plus": 0.035, "Pro": 0.02}[plan]

    tenure = 0
    active = True
    churn_month_idx = None
    m_idx = signup_month_idx
    while active and m_idx < MONTHS:
        tenure += 1
        hazard = base_hazard * max(0.5, 1 - tenure * 0.02)
        if np.random.random() < hazard:
            active = False
            churn_month_idx = m_idx
        m_idx += 1

    customers.append({
        "customer_id": cust_id,
        "signup_date": signup_date,
        "plan": plan,
        "mrr": mrr,
        "churn_date": month_add(START, churn_month_idx) if churn_month_idx else None,
        "is_active": churn_month_idx is None,
    })

customers_df = pd.DataFrame(customers)

billing_rows = []
for _, c in customers_df.iterrows():
    signup_idx = (c["signup_date"].year - START.year) * 12 + (c["signup_date"].month - START.month)
    if c["churn_date"] is not None:
        end_idx = (c["churn_date"].year - START.year) * 12 + (c["churn_date"].month - START.month)
    else:
        end_idx = MONTHS
    for m_idx in range(signup_idx, end_idx):
        billing_rows.append({
            "customer_id": c["customer_id"],
            "billing_month": month_add(START, m_idx),
            "plan": c["plan"],
            "mrr": c["mrr"],
        })

billing_df = pd.DataFrame(billing_rows)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
customers_df.to_csv(os.path.join(SCRIPT_DIR, "customers.csv"), index=False)
billing_df.to_csv(os.path.join(SCRIPT_DIR, "billing_events.csv"), index=False)

active_count = customers_df["is_active"].sum()
churned_count = (~customers_df["is_active"]).sum()
print(f"Customers: {len(customers_df)}  |  Billing rows: {len(billing_df)}")
print(f"Active: {active_count}  |  Churned: {churned_count}")
