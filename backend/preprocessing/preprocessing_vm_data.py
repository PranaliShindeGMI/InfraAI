# backend/preprocessing/preprocessing_vm_data.py
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

def preprocess_vm_data(df: pd.DataFrame, save_to_file: bool = True, output_filename: str = 'processed_vm_data.csv') -> pd.DataFrame:
    """
    Preprocess VM instance data:
    - Convert timestamp to datetime
    - Handle missing values
    - Convert scientific notation
    - Create derived features
    - Aggregate by date AND instance_id (keep instances separate)
    - Optionally save to processed folder
    
    Args:
        df: Input DataFrame with raw VM data
        save_to_file: Whether to save the processed data to CSV (default: True)
        output_filename: Name of the output CSV file (default: 'processed_vm_data.csv')
    
    Returns:
        Processed DataFrame with each instance kept separate
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Handle memory_used_bytes scientific notation
    df['memory_used_gb'] = df['memory_used_bytes'] / (1024**3)  # Convert to GB
    
    # Create efficiency metrics
    df['cost_per_cpu'] = df['cost_usd'] / (df['cpu_utilization'] + 0.001)  # Avoid division by zero
    df['network_total_bytes'] = df['ingress_bytes'] + df['egress_bytes']
    df['disk_total_bytes'] = df['disk_read_bytes'] + df['disk_write_bytes']
    
    # Extract instance ID from resource_global_name
    df['instance_id'] = df['resource_global_name'].str.extract(r'/instances/(\d+)')
    
    # Extract zone from resource_global_name
    df['zone'] = df['resource_global_name'].str.extract(r'/zones/([^/]+)/')
    
    # Extract project from resource_global_name
    df['project_id'] = df['resource_global_name'].str.extract(r'/projects/([^/]+)/')
    
    # Categorize SKU types
    df['sku_category'] = df['sku_description'].apply(categorize_sku)
    
    # Group and aggregate by date AND instance_id to keep instances separate
    daily = df.groupby(['date', 'instance_id', 'project_id', 'zone'], dropna=False).agg({
        'cpu_utilization': ['mean', 'std', 'min', 'max'],
        'memory_used_gb': ['mean', 'std', 'min', 'max'],
        'disk_read_bytes': ['sum', 'mean'],
        'disk_write_bytes': ['sum', 'mean'],
        'ingress_bytes': ['sum', 'mean'],
        'egress_bytes': ['sum', 'mean'],
        'network_total_bytes': ['sum', 'mean'],
        'disk_total_bytes': ['sum', 'mean'],
        'uptime_fraction': ['mean', 'min', 'max'],
        'cost_usd': ['sum', 'mean', 'max'],
        'cost_per_cpu': 'mean',
        'sku_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common SKU category
    }).reset_index()
    
    # Flatten column names
    daily.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in daily.columns.values]
    
    # Convert date column to string for JSON serialization
    daily['date'] = daily['date'].astype(str)
    
    # Sort by date and instance_id for better readability
    daily = daily.sort_values(['date', 'instance_id']).reset_index(drop=True)
    
    # Save to processed folder if requested
    if save_to_file:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        output_path = os.path.join(output_dir, output_filename)
        daily.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
    
    return daily

def categorize_sku(sku_description: str) -> str:
    """Categorize SKU descriptions into broad categories"""
    sku_lower = sku_description.lower()
    if 'vm state' in sku_lower or 'ssd' in sku_lower:
        return 'Storage'
    elif 'network' in sku_lower and 'inter zone' in sku_lower:
        return 'Network_InterZone'
    elif 'network' in sku_lower and 'intra zone' in sku_lower:
        return 'Network_IntraZone'
    elif 'network' in sku_lower and 'google services' in sku_lower:
        return 'Network_GoogleServices'
    elif 'network' in sku_lower:
        return 'Network_Other'
    else:
        return 'Other'