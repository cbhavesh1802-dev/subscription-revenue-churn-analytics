import sqlite3
import pandas as pd
import os

DATA_DIR = "/home/claude/projects/02-subscription-churn-analytics/data"
DB_PATH = f"{DATA_DIR}/subscription_analytics.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
billing = pd.read_csv(f"{DATA_DIR}/billing_events.csv")

conn = sqlite3.connect(DB_PATH)
customers.to_sql("customers", conn, index=False)
billing.to_sql("billing_events", conn, index=False)

conn.execute("CREATE INDEX idx_cust ON billing_events(customer_id);")
conn.execute("CREATE INDEX idx_month ON billing_events(billing_month);")
conn.commit()
conn.close()

print(f"Loaded customers({len(customers)}) + billing_events({len(billing)}) into {DB_PATH}")
