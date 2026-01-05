import os
import argparse
import pandas as pd
import numpy as np
import json
from math import sqrt

KNOWN_FORECAST_TO_TRUTH = {
    "cpu_utilization_forecast": "cpu_utilization_mean",
    "cost_usd_forecast": "cost_usd_sum",
    "disk_total_bytes_forecast": "disk_total_bytes_sum",
    "network_total_bytes_forecast": "network_total_bytes_sum",
}

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        return float('nan')
    return (np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])).mean() * 100.0

def compute_metrics(series_true, series_pred):
    series_true = np.array(series_true, dtype=float)
    series_pred = np.array(series_pred, dtype=float)
    return {
        'mae': float(mae(series_true, series_pred)),
        'rmse': float(rmse(series_true, series_pred)),
        'mape': float(mape(series_true, series_pred)),
        'n': int(len(series_true))
    }

def load_dataframe(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def guess_truth_column(forecast_col, truth_columns):
    """
    Try to find the most likely ground-truth column name for a given forecast column.
    Heuristics:
    1) Use KNOWN_FORECAST_TO_TRUTH mapping if present.
    2) Strip trailing '_forecast' and try exact matches with common suffixes:
       - base + '_mean', base + '_sum', base + '_total', base
    3) Try substring matching if still not found.
    Returns the truth column name or None.
    """
    if forecast_col in KNOWN_FORECAST_TO_TRUTH:
        return KNOWN_FORECAST_TO_TRUTH[forecast_col]

    if forecast_col.endswith('_forecast'):
        base = forecast_col[:-len('_forecast')]
    else:
        base = forecast_col

    candidates = [
        base + "_mean",
        base + "_sum",
        base + "_total",
        base + "_bytes_sum",
        base + "_bytes_total",
        base
    ]
    for c in candidates:
        if c in truth_columns:
            return c

    if len(base) >= 3:
        for tc in truth_columns:
            if base in tc:
                return tc

    return None

def evaluate_period(forecasts_df, truth_df, period_df, period_name):
    key = ['date', 'instance_id']
    eval_keys = period_df[key].drop_duplicates()
    merged = pd.merge(eval_keys, forecasts_df, on=key, how='left')
    merged = pd.merge(merged, truth_df, on=key, how='left', suffixes=('_forecast_src', '_truth'))

    report = {'period': period_name, 'rows_requested': len(eval_keys), 'rows_matched': 0, 'metrics': {}}

    if merged.empty:
        report['note'] = 'No rows to evaluate'
        return report

    forecast_cols = [c for c in forecasts_df.columns if c.endswith('_forecast')]
    truth_columns = list(truth_df.columns)

    rows_matched_total = 0
    for fc in forecast_cols:
        truth_col = guess_truth_column(fc, truth_columns)
        if truth_col is None:
            report['metrics'][fc] = {'note': f"No matching ground-truth column found for forecast column '{fc}'."}
            continue

        sel = merged[[fc, truth_col]].dropna()
        if sel.empty:
            report['metrics'][truth_col] = {'note': f"No overlapping forecast & truth rows after dropna for '{fc}' -> '{truth_col}'."}
            continue

        metrics = compute_metrics(sel[truth_col].astype(float), sel[fc].astype(float))
        report['metrics'][truth_col] = metrics
        rows_matched_total += len(sel)

    report['rows_matched'] = int(rows_matched_total)
    return report

def evaluate_all(forecast_path=None, processed_path=None, test_path=None, validation_path=None,
                 output_json=None, year=None, only_validation=False):
    base = os.path.dirname(__file__)
    if forecast_path is None:
        forecast_path = os.path.join(base, 'data', 'forecast', 'vm_forecasts.csv')
    if processed_path is None:
        processed_path = os.path.join(base, '..', '..', 'data', 'processed', 'processed_vm_data.csv')
    if test_path is None:
        test_path = os.path.join(base, 'data', 'test', 'test_vm_data.csv')
    if validation_path is None:
        validation_path = os.path.join(base, 'data', 'validation', 'validation_vm_data.csv')
    if output_json is None:
        output_json = os.path.join(base, 'data', 'forecast', 'evaluation_report.json')

    forecasts_df = load_dataframe(forecast_path)
    truth_df = load_dataframe(processed_path)

    if year is None:
        year = int(truth_df['date'].dt.year.max())

    val_start = pd.Timestamp(year=year, month=10, day=1)
    val_end = pd.Timestamp(year=year, month=12, day=31)

    reports = []

    if os.path.exists(validation_path):
        val_df = load_dataframe(validation_path)
        val_df = val_df[(val_df['date'] >= val_start) & (val_df['date'] <= val_end)].copy()
        if val_df.empty:
            reports.append({'period': 'validation', 'note': f'Validation file found but contains no rows in {val_start.date()} to {val_end.date()}.'})
        else:
            reports.append(evaluate_period(forecasts_df, truth_df, val_df, 'validation'))
    else:
        val_df = truth_df[(truth_df['date'] >= val_start) & (truth_df['date'] <= val_end)].copy()
        if val_df.empty:
            reports.append({'period': 'validation', 'note': f'No ground-truth rows found in {val_start.date()} to {val_end.date()} (cannot evaluate validation).'})
        else:
            reports.append(evaluate_period(forecasts_df, truth_df, val_df, 'validation'))

    if not only_validation:
        if os.path.exists(test_path):
            test_df = load_dataframe(test_path)
            reports.append(evaluate_period(forecasts_df, truth_df, test_df, 'test'))
        else:
            reports.append({'period': 'test', 'note': f'{test_path} not found'})

    # Save report JSON
    try:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"Evaluation saved to {output_json}")
    except Exception as e:
        print(f"Could not save evaluation JSON: {e}")

    # Console summary
    for r in reports:
        print("\n---", r.get('period', 'unknown').upper(), "---")
        if 'note' in r:
            print(r['note'])
            continue
        print(f"Rows requested: {r.get('rows_requested', 'n/a')}")
        print(f"Rows matched (total across metrics): {r.get('rows_matched', 'n/a')}")
        for k, v in r.get('metrics', {}).items():
            if 'note' in v:
                print(f"{k}: {v['note']}")
            else:
                print(f"{k}: n={v['n']}, MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, MAPE={v['mape']:.2f}%")

    return reports

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Evaluate forecasts against test/validation CSVs.")
    parser.add_argument('--forecast', help='Path to forecasts CSV', default=None)
    parser.add_argument('--processed', help='Path to processed ground-truth CSV', default=None)
    parser.add_argument('--test', help='Path to test CSV', default=None)
    parser.add_argument('--validation', help='Path to validation CSV', default=None)
    parser.add_argument('--output', help='Path to output JSON report', default=None)
    parser.add_argument('--year', help='Year to evaluate (defaults to latest year in processed data)', type=int, default=None)
    parser.add_argument('--only-validation', help='Evaluate only validation (Oct-Dec) and skip test', action='store_true')
    args = parser.parse_args()

    evaluate_all(forecast_path=args.forecast,
                 processed_path=args.processed,
                 test_path=args.test,
                 validation_path=args.validation,
                 output_json=args.output,
                 year=args.year,
                 only_validation=args.only_validation)

if __name__ == "__main__":
    parse_args_and_run()