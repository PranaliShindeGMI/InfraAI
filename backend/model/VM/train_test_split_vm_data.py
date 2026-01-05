import pandas as pd
import os
from datetime import datetime

def train_test_split_vm_data(year: int = None, test_ratio: float = 0.25):
    """
    Split VM data into train, test and validation sets.

    Behavior:
    - Uses the latest year in the data if `year` is None.
    - Validation: last 3 months of the year (October 1 to December 31).
    - Train/Test pool: Jan 1 to Sep 30 of the same year.
      That pool is split chronologically into Train (75%) and Test (25%) by default.
    - Saves CSVs to `data/train/train_vm_data.csv`, `data/test/test_vm_data.csv`,
      and `data/validation/validation_vm_data.csv` under this folder.

    Args:
        year: explicit year to use for splitting. If None, uses latest year available in the data.
        test_ratio: proportion of the pool used for test (default 0.25 -> 3:1 split).
    Returns:
        train_df, test_df, validation_df
    """
    try:
        # Paths
        processed_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'processed_vm_data.csv')
        output_dir = os.path.join(os.path.dirname(__file__), 'data')
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        val_dir = os.path.join(output_dir, 'validation')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Load data
        df = pd.read_csv(processed_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        if year is None:
            year = int(df['date'].dt.year.max())

        val_start = pd.Timestamp(year=year, month=10, day=1)
        val_end = pd.Timestamp(year=year, month=12, day=31)

        # Validation set: dates in Oct-Dec of the chosen year
        validation_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()

        # Pool for train/test: Jan 1 to Sep 30 of the chosen year
        pool_start = pd.Timestamp(year=year, month=1, day=1)
        pool_end = pd.Timestamp(year=year, month=9, day=30)
        pool_df = df[(df['date'] >= pool_start) & (df['date'] <= pool_end)].copy()

        if pool_df.empty:
            pool_df = df[df['date'] < val_start].copy()

        if pool_df.empty:
            raise RuntimeError(f"No data available for training/testing pool for year {year} (pool empty).")

        # Chronological split of pool into train/test
        pool_df = pool_df.sort_values('date')
        split_idx = int(len(pool_df) * (1 - test_ratio))
        if split_idx < 1 or split_idx >= len(pool_df):
            split_idx = max(1, min(len(pool_df) - 1, split_idx))

        train_df = pool_df.iloc[:split_idx].copy()
        test_df = pool_df.iloc[split_idx:].copy()

        # Save CSVs
        train_path = os.path.join(train_dir, 'train_vm_data.csv')
        test_path = os.path.join(test_dir, 'test_vm_data.csv')
        val_path = os.path.join(val_dir, 'validation_vm_data.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        validation_df.to_csv(val_path, index=False)

        # Logging summary
        print("--- Split Summary ---")
        print(f"Year used: {year}")
        print(f"Training: {len(train_df)} rows from {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Testing:  {len(test_df)} rows from {test_df['date'].min()} to {test_df['date'].max()}")
        if not validation_df.empty:
            print(f"Validation:{len(validation_df)} rows from {validation_df['date'].min()} to {validation_df['date'].max()}")
        else:
            print("Validation: 0 rows (no Oct-Dec data found for year)")

        return train_df, test_df, validation_df

    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None


if __name__ == "__main__":
    try:
        train_test_split_vm_data()
    except Exception as e:
        print(f"An error occurred during script execution: {e}")