# backend/analysis/analysis_vm_data.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import json

def analyze_vm_data(preprocessed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on VM data including:
    - Overall statistics across all instances
    - Per-instance analysis
    - Correlation analysis
    - Trend analysis
    - Anomaly detection
    - Cost efficiency metrics
    """
    
    # Preprocess data first (now keeps instances separate)

    
    analysis_results = {}
    
    # Add instance summary
    analysis_results['instance_summary'] = get_instance_summary(preprocessed_df)
    
    # 1. Overall Descriptive Statistics (aggregated across all instances)
    analysis_results['overall_stats'] = get_descriptive_stats(preprocessed_df)
    
    # 2. Per-Instance Analysis
    analysis_results['per_instance_analysis'] = get_per_instance_analysis(preprocessed_df)
    
    # 3. Correlation Analysis (across all instances)
    analysis_results['correlation_analysis'] = get_correlation_analysis(preprocessed_df)
    
    # 4. Trend Analysis (per instance and overall)
    analysis_results['trend_analysis'] = get_trend_analysis(preprocessed_df)
    
    # 5. Cost Analysis (overall and per instance)
    analysis_results['cost_analysis'] = get_cost_analysis(preprocessed_df)
    
    # 6. Resource Utilization Insights
    analysis_results['utilization_insights'] = get_utilization_insights(preprocessed_df)
    
    # 7. Anomaly Detection (per instance)
    analysis_results['anomalies'] = detect_anomalies(preprocessed_df)

    
    
    output_dir = "backend/data/analysis"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "vm_analysis_results.json")
    with open(file_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    return analysis_results

def get_instance_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary of instances in the dataset"""
    summary = {}
    
    if 'instance_id' in df.columns:
        instances = df['instance_id'].unique()
        summary['total_instances'] = int(len(instances))
        summary['instance_ids'] = sorted([str(i) for i in instances])
        
        # Count records per instance
        instance_counts = df['instance_id'].value_counts().to_dict()
        summary['records_per_instance'] = {str(k): int(v) for k, v in instance_counts.items()}
    
    if 'date' in df.columns:
        summary['date_range'] = {
            'start': str(df['date'].min()),
            'end': str(df['date'].max()),
            'total_days': int(df['date'].nunique())
        }
    
    return summary

def get_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate descriptive statistics for key metrics across all instances"""
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

def get_per_instance_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze each instance separately"""
    if 'instance_id' not in df.columns:
        return {}
    
    per_instance = {}
    
    for instance_id in df['instance_id'].unique():
        instance_df = df[df['instance_id'] == instance_id].copy()
        instance_key = str(instance_id)
        
        per_instance[instance_key] = {
            'total_records': int(len(instance_df)),
            'date_range': {
                'start': str(instance_df['date'].min()),
                'end': str(instance_df['date'].max())
            }
        }
        
        # Key metrics averages for this instance
        if 'cpu_utilization_mean' in instance_df.columns:
            per_instance[instance_key]['avg_cpu_utilization'] = float(instance_df['cpu_utilization_mean'].mean())
        
        if 'memory_used_gb_mean' in instance_df.columns:
            per_instance[instance_key]['avg_memory_gb'] = float(instance_df['memory_used_gb_mean'].mean())
        
        if 'cost_usd_sum' in instance_df.columns:
            cost_data = instance_df['cost_usd_sum'].dropna()
            if len(cost_data) > 0:
                per_instance[instance_key]['total_cost'] = float(cost_data.sum())
                per_instance[instance_key]['avg_daily_cost'] = float(cost_data.mean())
        
        if 'uptime_fraction_mean' in instance_df.columns:
            per_instance[instance_key]['avg_uptime_fraction'] = float(instance_df['uptime_fraction_mean'].mean())
        
        if 'network_total_bytes_sum' in instance_df.columns:
            per_instance[instance_key]['total_network_gb'] = float(instance_df['network_total_bytes_sum'].sum() / (1024**3))
        
        # Zone and project info
        if 'zone' in instance_df.columns:
            per_instance[instance_key]['zone'] = str(instance_df['zone'].iloc[0])
        
        if 'project_id' in instance_df.columns:
            per_instance[instance_key]['project_id'] = str(instance_df['project_id'].iloc[0])
    
    return per_instance

def get_correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform correlation analysis on key metrics across all instances
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
    
    if len(available_cols) < 2:
        return {}
    
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
    """Analyze trends over time - both overall and per instance"""
    trends = {}
    
    # Overall trends (aggregate all instances by date)
    if 'date' in df.columns:
        date_aggregated = df.groupby('date').agg({
            'cpu_utilization_mean': 'mean',
            'memory_used_gb_mean': 'mean',
            'cost_usd_sum': 'sum',
            'uptime_fraction_mean': 'mean',
            'network_total_bytes_sum': 'sum'
        }).reset_index().sort_values('date')
        
        if len(date_aggregated) >= 2:
            first_row = date_aggregated.iloc[0]
            last_row = date_aggregated.iloc[-1]
            
            trends['overall'] = {
                'date_range': {
                    'start_date': str(first_row['date']),
                    'end_date': str(last_row['date'])
                }
            }
            
            metrics = {
                'cpu_utilization_mean': 'CPU Utilization',
                'memory_used_gb_mean': 'Memory Usage (GB)',
                'cost_usd_sum': 'Daily Cost',
                'uptime_fraction_mean': 'Uptime Fraction',
                'network_total_bytes_sum': 'Total Network Traffic'
            }
            
            for metric, label in metrics.items():
                if metric in date_aggregated.columns:
                    first_val = first_row[metric]
                    last_val = last_row[metric]
                    if first_val != 0 and not pd.isna(first_val) and not pd.isna(last_val):
                        pct_change = ((last_val - first_val) / first_val) * 100
                        trends['overall'][label] = {
                            'start_value': float(first_val),
                            'end_value': float(last_val),
                            'percent_change': float(pct_change),
                            'trend': 'increasing' if pct_change > 5 else ('decreasing' if pct_change < -5 else 'stable')
                        }
    
    # Per-instance trends - Aggregated metrics across date range
    if 'instance_id' in df.columns:
        trends['per_instance'] = {}
        
        for instance_id in df['instance_id'].unique():
            instance_df = df[df['instance_id'] == instance_id].sort_values('date')
            
            if len(instance_df) >= 1:
                instance_key = str(instance_id)
                trends['per_instance'][instance_key] = {
                    'date_range': {
                        'start_date': str(instance_df['date'].min()),
                        'end_date': str(instance_df['date'].max()),
                        'total_days': int(len(instance_df))
                    }
                }
                
                # CPU utilization
                if 'cpu_utilization_mean' in instance_df.columns:
                    cpu_data = instance_df['cpu_utilization_mean'].dropna()
                    if len(cpu_data) > 0:
                        trends['per_instance'][instance_key]['cpu_utilization'] = {
                            'mean': float(cpu_data.mean()),
                            'median': float(cpu_data.median()),
                            'min': float(cpu_data.min()),
                            'max': float(cpu_data.max()),
                            'std': float(cpu_data.std()) if len(cpu_data) > 1 else 0.0,
                            'trend': get_trend_direction(instance_df, 'cpu_utilization_mean')
                        }
                
                # Memory usage
                if 'memory_used_gb_mean' in instance_df.columns:
                    mem_data = instance_df['memory_used_gb_mean'].dropna()
                    if len(mem_data) > 0:
                        trends['per_instance'][instance_key]['memory_usage_gb'] = {
                            'mean': float(mem_data.mean()),
                            'median': float(mem_data.median()),
                            'min': float(mem_data.min()),
                            'max': float(mem_data.max()),
                            'std': float(mem_data.std()) if len(mem_data) > 1 else 0.0,
                            'trend': get_trend_direction(instance_df, 'memory_used_gb_mean')
                        }
                
                # Cost
                if 'cost_usd_sum' in instance_df.columns:
                    cost_data = instance_df['cost_usd_sum'].dropna()
                    if len(cost_data) > 0:
                        trends['per_instance'][instance_key]['cost'] = {
                            'total': float(cost_data.sum()),
                            'mean_daily': float(cost_data.mean()),
                            'median_daily': float(cost_data.median()),
                            'min_daily': float(cost_data.min()),
                            'max_daily': float(cost_data.max()),
                            'std': float(cost_data.std()) if len(cost_data) > 1 else 0.0,
                            'trend': get_trend_direction(instance_df, 'cost_usd_sum')
                        }
                
                # Uptime fraction
                if 'uptime_fraction_mean' in instance_df.columns:
                    uptime_data = instance_df['uptime_fraction_mean'].dropna()
                    if len(uptime_data) > 0:
                        trends['per_instance'][instance_key]['uptime_fraction'] = {
                            'mean': float(uptime_data.mean()),
                            'median': float(uptime_data.median()),
                            'min': float(uptime_data.min()),
                            'max': float(uptime_data.max()),
                            'std': float(uptime_data.std()) if len(uptime_data) > 1 else 0.0,
                            'trend': get_trend_direction(instance_df, 'uptime_fraction_mean')
                        }
                
                # Network traffic
                if 'network_total_bytes_sum' in instance_df.columns:
                    network_data = instance_df['network_total_bytes_sum'].dropna()
                    if len(network_data) > 0:
                        trends['per_instance'][instance_key]['network_traffic'] = {
                            'total_gb': float(network_data.sum() / (1024**3)),
                            'mean_daily_gb': float(network_data.mean() / (1024**3)),
                            'median_daily_gb': float(network_data.median() / (1024**3)),
                            'min_daily_gb': float(network_data.min() / (1024**3)),
                            'max_daily_gb': float(network_data.max() / (1024**3)),
                            'trend': get_trend_direction(instance_df, 'network_total_bytes_sum')
                        }
                
                # Disk read
                if 'disk_read_bytes_sum' in instance_df.columns:
                    disk_read_data = instance_df['disk_read_bytes_sum'].dropna()
                    if len(disk_read_data) > 0:
                        trends['per_instance'][instance_key]['disk_read'] = {
                            'total_gb': float(disk_read_data.sum() / (1024**3)),
                            'mean_daily_gb': float(disk_read_data.mean() / (1024**3)),
                            'median_daily_gb': float(disk_read_data.median() / (1024**3)),
                            'trend': get_trend_direction(instance_df, 'disk_read_bytes_sum')
                        }
                
                # Disk write
                if 'disk_write_bytes_sum' in instance_df.columns:
                    disk_write_data = instance_df['disk_write_bytes_sum'].dropna()
                    if len(disk_write_data) > 0:
                        trends['per_instance'][instance_key]['disk_write'] = {
                            'total_gb': float(disk_write_data.sum() / (1024**3)),
                            'mean_daily_gb': float(disk_write_data.mean() / (1024**3)),
                            'median_daily_gb': float(disk_write_data.median() / (1024**3)),
                            'trend': get_trend_direction(instance_df, 'disk_write_bytes_sum')
                        }
                
                # Network ingress
                if 'ingress_bytes_sum' in instance_df.columns:
                    ingress_data = instance_df['ingress_bytes_sum'].dropna()
                    if len(ingress_data) > 0:
                        trends['per_instance'][instance_key]['network_ingress'] = {
                            'total_gb': float(ingress_data.sum() / (1024**3)),
                            'mean_daily_gb': float(ingress_data.mean() / (1024**3)),
                            'trend': get_trend_direction(instance_df, 'ingress_bytes_sum')
                        }
                
                # Network egress
                if 'egress_bytes_sum' in instance_df.columns:
                    egress_data = instance_df['egress_bytes_sum'].dropna()
                    if len(egress_data) > 0:
                        trends['per_instance'][instance_key]['network_egress'] = {
                            'total_gb': float(egress_data.sum() / (1024**3)),
                            'mean_daily_gb': float(egress_data.mean() / (1024**3)),
                            'trend': get_trend_direction(instance_df, 'egress_bytes_sum')
                        }
                
                # Cost efficiency
                if 'cost_per_cpu_mean' in instance_df.columns:
                    cost_eff_data = instance_df['cost_per_cpu_mean'].dropna()
                    if len(cost_eff_data) > 0:
                        trends['per_instance'][instance_key]['cost_efficiency'] = {
                            'mean_cost_per_cpu': float(cost_eff_data.mean()),
                            'median_cost_per_cpu': float(cost_eff_data.median()),
                            'min_cost_per_cpu': float(cost_eff_data.min()),
                            'max_cost_per_cpu': float(cost_eff_data.max()),
                            'trend': get_trend_direction(instance_df, 'cost_per_cpu_mean')
                        }
    
    return trends

def get_trend_direction(df: pd.DataFrame, metric: str) -> str:
    """
    Determine trend direction using linear regression
    Returns: 'increasing', 'decreasing', or 'stable'
    """
    if len(df) < 2:
        return 'insufficient_data'
    
    metric_data = df[metric].dropna()
    if len(metric_data) < 2:
        return 'insufficient_data'
    
    # Simple approach: compare first half average to second half average
    mid_point = len(metric_data) // 2
    first_half_mean = metric_data.iloc[:mid_point].mean()
    second_half_mean = metric_data.iloc[mid_point:].mean()
    
    if first_half_mean == 0:
        return 'stable'
    
    pct_change = ((second_half_mean - first_half_mean) / first_half_mean) * 100
    
    if pct_change > 5:
        return 'increasing'
    elif pct_change < -5:
        return 'decreasing'
    else:
        return 'stable'

def get_cost_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze cost patterns and efficiency - overall and per instance"""
    cost_analysis = {}
    
    # Overall cost analysis
    if 'cost_usd_sum' in df.columns:
        cost_data = df['cost_usd_sum'].dropna()
        
        if len(cost_data) > 0:
            cost_analysis['overall'] = {
                'total_cost': float(cost_data.sum()),
                'average_daily_cost_per_instance': float(cost_data.mean()),
                'max_daily_cost': float(cost_data.max()),
                'min_daily_cost': float(cost_data.min()),
                'cost_std_deviation': float(cost_data.std()) if len(cost_data) > 1 else 0.0
            }
            
            # Projected monthly cost
            cost_analysis['overall']['projected_monthly_cost'] = cost_analysis['overall']['average_daily_cost_per_instance'] * 30
            
            # Cost efficiency (cost per CPU utilization)
            if 'cost_per_cpu_mean' in df.columns:
                cpu_cost_data = df['cost_per_cpu_mean'].dropna()
                if len(cpu_cost_data) > 0:
                    cost_analysis['overall']['average_cost_per_cpu_utilization'] = float(cpu_cost_data.mean())
    
    # Per-instance cost analysis
    if 'instance_id' in df.columns and 'cost_usd_sum' in df.columns:
        cost_analysis['per_instance'] = {}
        
        for instance_id in df['instance_id'].unique():
            instance_df = df[df['instance_id'] == instance_id]
            instance_cost_data = instance_df['cost_usd_sum'].dropna()
            
            if len(instance_cost_data) > 0:
                instance_key = str(instance_id)
                cost_analysis['per_instance'][instance_key] = {
                    'total_cost': float(instance_cost_data.sum()),
                    'avg_daily_cost': float(instance_cost_data.mean()),
                    'max_daily_cost': float(instance_cost_data.max()),
                    'min_daily_cost': float(instance_cost_data.min())
                }
    
    return cost_analysis

def get_utilization_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate insights about resource utilization - overall and per instance with all metrics"""
    insights = {}
    
    # Overall utilization (across all instances)
    insights['overall'] = {}
    
    # CPU utilization analysis
    if 'cpu_utilization_mean' in df.columns:
        cpu_data = df['cpu_utilization_mean'].dropna()
        if len(cpu_data) > 0:
            cpu_mean = cpu_data.mean()
            insights['overall']['cpu'] = {
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
            insights['overall']['memory'] = {
                'average_memory_gb': float(mem_mean),
                'max_memory_gb': float(df['memory_used_gb_max'].max() if 'memory_used_gb_max' in df.columns else mem_mean),
                'min_memory_gb': float(df['memory_used_gb_min'].min() if 'memory_used_gb_min' in df.columns else mem_mean)
            }
    
    # Network utilization
    if 'network_total_bytes_sum' in df.columns:
        network_data = df['network_total_bytes_sum'].dropna()
        if len(network_data) > 0:
            network_total = network_data.sum()
            insights['overall']['network'] = {
                'total_network_bytes': float(network_total),
                'total_network_gb': float(network_total / (1024**3)),
                'average_daily_network_gb': float(network_data.mean() / (1024**3))
            }
    
    # Uptime analysis
    if 'uptime_fraction_mean' in df.columns:
        uptime_data = df['uptime_fraction_mean'].dropna()
        if len(uptime_data) > 0:
            uptime_mean = uptime_data.mean()
            insights['overall']['uptime'] = {
                'average_uptime_fraction': float(uptime_mean),
                'uptime_percentage': float(uptime_mean * 100),
                'reliability_rating': (
                    'excellent' if uptime_mean > 0.95 else
                    'good' if uptime_mean > 0.85 else
                    'needs improvement'
                )
            }
    
    # Per-instance utilization insights - ALL METRICS
    if 'instance_id' in df.columns:
        insights['per_instance'] = {}
        
        for instance_id in df['instance_id'].unique():
            instance_df = df[df['instance_id'] == instance_id]
            instance_key = str(instance_id)
            insights['per_instance'][instance_key] = {}
            
            # CPU for this instance
            if 'cpu_utilization_mean' in instance_df.columns:
                cpu_data = instance_df['cpu_utilization_mean'].dropna()
                if len(cpu_data) > 0:
                    cpu_mean = cpu_data.mean()
                    insights['per_instance'][instance_key]['cpu'] = {
                        'average': float(cpu_mean),
                        'max': float(instance_df['cpu_utilization_max'].max() if 'cpu_utilization_max' in instance_df.columns else cpu_mean),
                        'min': float(instance_df['cpu_utilization_min'].min() if 'cpu_utilization_min' in instance_df.columns else cpu_mean),
                        'std': float(cpu_data.std()) if len(cpu_data) > 1 else 0.0,
                        'category': (
                            'over-utilized' if cpu_mean > 0.8 else
                            'well-utilized' if cpu_mean > 0.5 else
                            'under-utilized'
                        )
                    }
            
            # Memory for this instance
            if 'memory_used_gb_mean' in instance_df.columns:
                mem_data = instance_df['memory_used_gb_mean'].dropna()
                if len(mem_data) > 0:
                    insights['per_instance'][instance_key]['memory'] = {
                        'average_gb': float(mem_data.mean()),
                        'max_gb': float(instance_df['memory_used_gb_max'].max() if 'memory_used_gb_max' in instance_df.columns else mem_data.max()),
                        'min_gb': float(instance_df['memory_used_gb_min'].min() if 'memory_used_gb_min' in instance_df.columns else mem_data.min()),
                        'std_gb': float(mem_data.std()) if len(mem_data) > 1 else 0.0
                    }
            
            # Network for this instance
            if 'network_total_bytes_sum' in instance_df.columns:
                network_data = instance_df['network_total_bytes_sum'].dropna()
                if len(network_data) > 0:
                    insights['per_instance'][instance_key]['network'] = {
                        'total_gb': float(network_data.sum() / (1024**3)),
                        'avg_daily_gb': float(network_data.mean() / (1024**3)),
                        'max_daily_gb': float(network_data.max() / (1024**3)),
                        'min_daily_gb': float(network_data.min() / (1024**3))
                    }
                    
                    # Breakdown by ingress/egress
                    if 'ingress_bytes_sum' in instance_df.columns and 'egress_bytes_sum' in instance_df.columns:
                        ingress_data = instance_df['ingress_bytes_sum'].dropna()
                        egress_data = instance_df['egress_bytes_sum'].dropna()
                        if len(ingress_data) > 0 and len(egress_data) > 0:
                            insights['per_instance'][instance_key]['network']['ingress_gb'] = float(ingress_data.sum() / (1024**3))
                            insights['per_instance'][instance_key]['network']['egress_gb'] = float(egress_data.sum() / (1024**3))
            
            # Disk I/O for this instance
            disk_metrics = {}
            if 'disk_read_bytes_sum' in instance_df.columns:
                disk_read_data = instance_df['disk_read_bytes_sum'].dropna()
                if len(disk_read_data) > 0:
                    disk_metrics['total_read_gb'] = float(disk_read_data.sum() / (1024**3))
                    disk_metrics['avg_daily_read_gb'] = float(disk_read_data.mean() / (1024**3))
            
            if 'disk_write_bytes_sum' in instance_df.columns:
                disk_write_data = instance_df['disk_write_bytes_sum'].dropna()
                if len(disk_write_data) > 0:
                    disk_metrics['total_write_gb'] = float(disk_write_data.sum() / (1024**3))
                    disk_metrics['avg_daily_write_gb'] = float(disk_write_data.mean() / (1024**3))
            
            if disk_metrics:
                insights['per_instance'][instance_key]['disk_io'] = disk_metrics
            
            # Uptime for this instance
            if 'uptime_fraction_mean' in instance_df.columns:
                uptime_data = instance_df['uptime_fraction_mean'].dropna()
                if len(uptime_data) > 0:
                    uptime_mean = uptime_data.mean()
                    insights['per_instance'][instance_key]['uptime'] = {
                        'average_fraction': float(uptime_mean),
                        'percentage': float(uptime_mean * 100),
                        'max_fraction': float(instance_df['uptime_fraction_max'].max() if 'uptime_fraction_max' in instance_df.columns else uptime_mean),
                        'min_fraction': float(instance_df['uptime_fraction_min'].min() if 'uptime_fraction_min' in instance_df.columns else uptime_mean),
                        'reliability_rating': (
                            'excellent' if uptime_mean > 0.95 else
                            'good' if uptime_mean > 0.85 else
                            'needs improvement'
                        )
                    }
            
            # Cost efficiency for this instance
            if 'cost_per_cpu_mean' in instance_df.columns:
                cost_eff_data = instance_df['cost_per_cpu_mean'].dropna()
                if len(cost_eff_data) > 0:
                    insights['per_instance'][instance_key]['cost_efficiency'] = {
                        'average_cost_per_cpu': float(cost_eff_data.mean()),
                        'max_cost_per_cpu': float(cost_eff_data.max()),
                        'min_cost_per_cpu': float(cost_eff_data.min())
                    }
    
    return insights

def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect anomalies in key metrics using IQR method - per instance"""
    anomalies = {}
    
    if 'instance_id' not in df.columns:
        return anomalies
    
    metrics_to_check = {
        'cost_usd_sum': 'Cost',
        'cpu_utilization_mean': 'CPU Utilization',
        'memory_used_gb_mean': 'Memory Usage',
        'network_total_bytes_sum': 'Network Traffic'
    }
    
    # Detect anomalies per instance
    for instance_id in df['instance_id'].unique():
        instance_df = df[df['instance_id'] == instance_id]
        instance_key = str(instance_id)
        instance_anomalies = {}
        
        for metric, label in metrics_to_check.items():
            if metric in instance_df.columns:
                metric_data = instance_df[metric].dropna()
                
                if len(metric_data) > 3:  # Need at least 4 data points for IQR
                    Q1 = metric_data.quantile(0.25)
                    Q3 = metric_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define outlier boundaries
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Find anomalies
                    anomaly_df = instance_df[(instance_df[metric] < lower_bound) | (instance_df[metric] > upper_bound)].copy()
                    
                    if len(anomaly_df) > 0:
                        instance_anomalies[label] = {
                            'count': int(len(anomaly_df)),
                            'dates': [str(date) for date in anomaly_df['date'].tolist()],
                            'values': [float(v) for v in anomaly_df[metric].tolist()],
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound)
                        }
        
        if instance_anomalies:
            anomalies[instance_key] = instance_anomalies
    
    return anomalies