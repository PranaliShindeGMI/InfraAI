# preprocessing/preprocessing.py
import pandas as pd

def preprocess_vm_data(df):
    """
    Preprocess VM instance data:
    - Convert timestamp to date
    - Aggregate by date:
        * cpu_utilization -> mean
        * memory_used_bytes -> mean
        * disk_read_bytes -> sum
        * disk_write_bytes -> sum
        * ingress_bytes -> sum
        * egress_bytes -> sum
        * uptime_fraction -> max
        * cost_usd -> sum
    """

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Group and aggregate
    daily = df.groupby('date').agg({
        'cpu_utilization': 'mean',
        'memory_used_bytes': 'mean',
        'disk_read_bytes': 'sum',
        'disk_write_bytes': 'sum',
        'ingress_bytes': 'sum',
        'egress_bytes': 'sum',
        'uptime_fraction': 'max',
        'cost_usd': 'sum'  # optional, but good to keep
    }).reset_index()

    return daily
