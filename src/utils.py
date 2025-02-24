"""utils.py — Config loading, logging, helpers."""
import os, yaml, datetime

def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
