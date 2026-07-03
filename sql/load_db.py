import sqlite3
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "subscription_analytics.db")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
billing = pd.read_csv(os.path.join(DATA_DIR, "billing_events.csv"))

conn = sqlite3.connect(DB_PATH)
customers.to_sql("customers", conn, index=False)
billing.to_sql("billing_events", conn, index=False)

conn.execute("CREATE INDEX idx_cust ON billing_events(customer_id);")
conn.execute("CREATE INDEX idx_month ON billing_events(billing_month);")
conn.commit()
conn.close()

print(f"Loaded customers({len(customers)}) + billing_events({len(billing)}) into {DB_PATH}")
