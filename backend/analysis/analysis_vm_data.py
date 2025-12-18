# backend/analysis/analysis_vm_data.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from backend.preprocessing import preprocess_vm_data

def analyze_vm_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on VM data including:
    - Descriptive statistics
    - Correlation analysis
    - Trend analysis
    - Anomaly detection
    - Cost efficiency metrics
    """
    
    # Preprocess data first
    preprocessed_df = preprocess_vm_data(df)
    
    analysis_results = {}
    
    # 1. Descriptive Statistics
    analysis_results['descriptive_stats'] = get_descriptive_stats(preprocessed_df)
    
    # 2. Correlation Analysis
    analysis_results['correlation_analysis'] = get_correlation_analysis(preprocessed_df)
    
    # 3. Trend Analysis
    analysis_results['trend_analysis'] = get_trend_analysis(preprocessed_df)
    
    # 4. Cost Analysis
    analysis_results['cost_analysis'] = get_cost_analysis(preprocessed_df)
    
    # 5. Resource Utilization Insights
    analysis_results['utilization_insights'] = get_utilization_insights(preprocessed_df)
    
    # 6. Anomaly Detection
    analysis_results['anomalies'] = detect_anomalies(preprocessed_df)
    
    return analysis_results

def get_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate descriptive statistics for key metrics"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = {}
    for col in numeric_cols:
        # Handle NaN values
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats[col] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75))
            }
    
    return stats

def get_correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform correlation analysis on key metrics
    """
    # Select relevant columns for correlation
    correlation_cols = [
        'cpu_utilization_mean', 
        'memory_used_gb_mean',
        'disk_read_bytes_sum',
        'disk_write_bytes_sum',
        'ingress_bytes_sum',
        'egress_bytes_sum',
        'network_total_bytes_sum',
        'uptime_fraction_mean',
        'cost_usd_sum',
        'cost_per_cpu_mean'
    ]
    
    # Filter only existing columns
    available_cols = [col for col in correlation_cols if col in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Replace NaN with 0 for JSON serialization
    corr_matrix = corr_matrix.fillna(0)
    
    # Find strong correlations (abs > 0.7, excluding diagonal)
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                strong_correlations.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': float(corr_value)
                })
    
    # Correlation with cost
    cost_correlations = []
    if 'cost_usd_sum' in corr_matrix.columns:
        cost_corr_series = corr_matrix['cost_usd_sum'].drop('cost_usd_sum').sort_values(ascending=False)
        for feature, corr_value in cost_corr_series.items():
            cost_correlations.append({
                'feature': feature,
                'correlation_with_cost': float(corr_value)
            })
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'strong_correlations': strong_correlations,
        'cost_correlations': cost_correlations
    }

def get_trend_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trends over time"""
    trends = {}
    
    # Sort by date
    df_sorted = df.sort_values('date')
    
    # Calculate percentage changes for key metrics
    if len(df_sorted) >= 2:
        first_row = df_sorted.iloc[0]
        last_row = df_sorted.iloc[-1]
        
        # Include date range in trends
        trends['date_range'] = {
            'start_date': str(first_row['date']),
            'end_date': str(last_row['date'])
        }
        
        metrics = {
            'cpu_utilization_mean': 'CPU Utilization',
            'memory_used_gb_mean': 'Memory Usage (GB)',
            'cost_usd_sum': 'Daily Cost',
            'uptime_fraction_mean': 'Uptime Fraction',
            'network_total_bytes_sum': 'Total Network Traffic'
        }
        
        for metric, label in metrics.items():
            if metric in df_sorted.columns:
                first_val = first_row[metric]
                last_val = last_row[metric]
                if first_val != 0 and not pd.isna(first_val) and not pd.isna(last_val):
                    pct_change = ((last_val - first_val) / first_val) * 100
                    trends[label] = {
                        'start_value': float(first_val),
                        'end_value': float(last_val),
                        'percent_change': float(pct_change),
                        'trend': 'increasing' if pct_change > 5 else ('decreasing' if pct_change < -5 else 'stable')
                    }
    
    return trends

def get_cost_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze cost patterns and efficiency"""
    cost_analysis = {}
    
    if 'cost_usd_sum' in df.columns:
        cost_data = df['cost_usd_sum'].dropna()
        
        if len(cost_data) > 0:
            # Total and average costs
            cost_analysis['total_cost'] = float(cost_data.sum())
            cost_analysis['average_daily_cost'] = float(cost_data.mean())
            cost_analysis['max_daily_cost'] = float(cost_data.max())
            cost_analysis['min_daily_cost'] = float(cost_data.min())
            cost_analysis['cost_std_deviation'] = float(cost_data.std()) if len(cost_data) > 1 else 0.0
            
            # Projected monthly cost
            cost_analysis['projected_monthly_cost'] = cost_analysis['average_daily_cost'] * 30
            
            # Cost efficiency (cost per CPU utilization)
            if 'cost_per_cpu_mean' in df.columns:
                cpu_cost_data = df['cost_per_cpu_mean'].dropna()
                if len(cpu_cost_data) > 0:
                    cost_analysis['average_cost_per_cpu_utilization'] = float(cpu_cost_data.mean())
            
            # Days with high costs (above 75th percentile)
            cost_threshold = cost_data.quantile(0.75)
            high_cost_days = df[df['cost_usd_sum'] > cost_threshold]
            cost_analysis['high_cost_days_count'] = int(len(high_cost_days))
            cost_analysis['high_cost_threshold'] = float(cost_threshold)
    
    return cost_analysis

def get_utilization_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate insights about resource utilization"""
    insights = {}
    
    # CPU utilization analysis
    if 'cpu_utilization_mean' in df.columns:
        cpu_data = df['cpu_utilization_mean'].dropna()
        if len(cpu_data) > 0:
            cpu_mean = cpu_data.mean()
            insights['cpu'] = {
                'average_utilization': float(cpu_mean),
                'utilization_category': (
                    'over-utilized' if cpu_mean > 0.8 else
                    'well-utilized' if cpu_mean > 0.5 else
                    'under-utilized'
                ),
                'max_utilization': float(df['cpu_utilization_max'].max() if 'cpu_utilization_max' in df.columns else cpu_mean),
                'min_utilization': float(df['cpu_utilization_min'].min() if 'cpu_utilization_min' in df.columns else cpu_mean)
            }
    
    # Memory utilization analysis
    if 'memory_used_gb_mean' in df.columns:
        mem_data = df['memory_used_gb_mean'].dropna()
        if len(mem_data) > 0:
            mem_mean = mem_data.mean()
            insights['memory'] = {
                'average_memory_gb': float(mem_mean),
                'max_memory_gb': float(df['memory_used_gb_max'].max() if 'memory_used_gb_max' in df.columns else mem_mean),
                'min_memory_gb': float(df['memory_used_gb_min'].min() if 'memory_used_gb_min' in df.columns else mem_mean)
            }
    
    # Network utilization
    if 'network_total_bytes_sum' in df.columns:
        network_data = df['network_total_bytes_sum'].dropna()
        if len(network_data) > 0:
            network_total = network_data.sum()
            insights['network'] = {
                'total_network_bytes': float(network_total),
                'total_network_gb': float(network_total / (1024**3)),
                'average_daily_network_gb': float(network_data.mean() / (1024**3))
            }
    
    # Uptime analysis
    if 'uptime_fraction_mean' in df.columns:
        uptime_data = df['uptime_fraction_mean'].dropna()
        if len(uptime_data) > 0:
            uptime_mean = uptime_data.mean()
            insights['uptime'] = {
                'average_uptime_fraction': float(uptime_mean),
                'uptime_percentage': float(uptime_mean * 100),
                'reliability_rating': (
                    'excellent' if uptime_mean > 0.95 else
                    'good' if uptime_mean > 0.85 else
                    'needs improvement'
                )
            }
    
    return insights

def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect anomalies in key metrics using IQR method"""
    anomalies = {}
    
    metrics_to_check = {
        'cost_usd_sum': 'Cost',
        'cpu_utilization_mean': 'CPU Utilization',
        'memory_used_gb_mean': 'Memory Usage',
        'network_total_bytes_sum': 'Network Traffic'
    }
    
    for metric, label in metrics_to_check.items():
        if metric in df.columns:
            metric_data = df[metric].dropna()
            
            if len(metric_data) > 0:
                Q1 = metric_data.quantile(0.25)
                Q3 = metric_data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier boundaries
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Find anomalies
                anomaly_df = df[(df[metric] < lower_bound) | (df[metric] > upper_bound)].copy()
                
                if len(anomaly_df) > 0:
                    anomalies[label] = {
                        'count': int(len(anomaly_df)),
                        'dates': [str(date) for date in anomaly_df['date'].tolist()],
                        'values': [float(v) for v in anomaly_df[metric].tolist()],
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
    
    return anomalies