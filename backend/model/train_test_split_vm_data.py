# backend/model/train_test_split_vm_data.py
import pandas as pd
import os

def train_test_split_vm_data(test_size: float = 0.2, split_by: str = 'time'):
    """
    Split processed VM data into train and test sets
    
    Args:
        test_size: Proportion of data for testing (default: 0.2 = 20%)
        split_by: 'time' for chronological split (recommended for time-series)
                  'random' for random split (not recommended for temporal data)
    
    Returns:
        train_df, test_df
    """
    # Load processed data
    processed_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_vm_data.csv')
    df = pd.read_csv(processed_path)
    
    # Convert date back to datetime for proper sorting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if split_by == 'time':
        # Time-based split (chronological)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        print(f"Time-based split:")
        print(f"  Training: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"  Testing:  {test_df['date'].min()} to {test_df['date'].max()}")
    else:
        # Random split (not recommended for time-series)
        train_df = df.sample(frac=(1-test_size), random_state=42)
        test_df = df[~df.index.isin(train_df.index)]
        print(f"Random split: {len(train_df)} train, {len(test_df)} test")
    
    # Save splits
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(output_dir, 'train', 'train_vm_data.csv')
    test_path = os.path.join(output_dir, 'test', 'test_vm_data.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    
    return train_df, test_df

if __name__ == "__main__":
    train_test_split_vm_data()