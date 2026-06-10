

def analyze_high_cost_contributors(meter_data, pu_gen_cost, high_cost_percentile):

    import pandas as pd
    import numpy as np
    import streamlit as st
    """
    Analyze meter/consumer contribution to high-cost periods
    
    Parameters:
    -----------
    meter_data : pd.DataFrame
        Shape (8760, 100) - Rows = time slots, Columns = meter numbers (100 meters)
    pu_gen_cost : pd.DataFrame or pd.Series
        Shape (8760, 1) or (8760,) - Hourly generation cost
    high_cost_percentile : int
        Percentile threshold to define high-cost periods (default: 75)
    
    Returns:
    --------
    pd.DataFrame : Analysis table with all meters and their metrics
    float : Cost threshold value
    int : Number of high-cost hours
    """
    
    # Ensure pu_gen_cost is a 1D array
    if isinstance(pu_gen_cost, pd.DataFrame):
        pu_cost = pu_gen_cost.iloc[:, 0].values  # Take first column
    elif isinstance(pu_gen_cost, pd.Series):
        pu_cost = pu_gen_cost.values
    else:
        pu_cost = np.array(pu_gen_cost)
    
    # Identify high-cost periods
    cost_threshold = np.percentile(pu_cost, high_cost_percentile)
    high_cost_mask = pu_cost >= cost_threshold
    high_cost_hours = high_cost_mask.sum()
    
    # Initialize results list
    results = []
    
    # Analyze each meter (each column)
    for meter_col in meter_data.columns:
        consumption = meter_data[meter_col].values
        
        # Overall metrics
        total_consumption = consumption.sum()
        peak_demand = consumption.max()
        avg_consumption = consumption.mean()
        
        # High-cost period metrics
        high_cost_consumption = consumption[high_cost_mask].sum()
        high_cost_avg = consumption[high_cost_mask].mean() if high_cost_hours > 0 else 0
        high_cost_peak = consumption[high_cost_mask].max() if high_cost_hours > 0 else 0
        
        # Contribution percentages
        pct_during_high_cost = (high_cost_consumption / total_consumption * 100) if total_consumption > 0 else 0
        pct_of_peak_during_high_cost = (high_cost_peak / peak_demand * 100) if peak_demand > 0 else 0
        
        # Cost impact
        total_cost_impact = (consumption * pu_cost).sum()
        high_cost_impact = (consumption[high_cost_mask] * pu_cost[high_cost_mask]).sum() if high_cost_hours > 0 else 0
        
        # Calculate risk score (0-100)
        risk_score = (
            pct_during_high_cost * 0.5 +           # Weight: consumption during high cost
            pct_of_peak_during_high_cost * 0.3 +   # Weight: peak during high cost
            (high_cost_impact / total_cost_impact * 100) * 0.2 if total_cost_impact > 0 else 0  # Weight: cost impact
        )
        
        results.append({
            'Meter': str(meter_col),  # Convert to string for display
            'Total_Consumption_MWh': round(total_consumption, 2),
            'Peak_Demand_MW': round(peak_demand, 2),
            'Avg_Consumption_MW': round(avg_consumption, 2),
            'High_Cost_Consumption_MWh': round(high_cost_consumption, 2),
            'Pct_During_High_Cost_%': round(pct_during_high_cost, 1),
            'Pct_of_Peak_in_High_Cost_%': round(pct_of_peak_during_high_cost, 1),
            'High_Cost_Avg_MW': round(high_cost_avg, 2),
            'High_Cost_Peak_MW': round(high_cost_peak, 2),
            'Total_Cost_Impact_₹': round(total_cost_impact, 0),
            'High_Cost_Impact_₹': round(high_cost_impact, 0),
            'Risk_Score': round(risk_score, 1)
        })
    
    # Create DataFrame and sort by risk score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Risk_Score', ascending=False).reset_index(drop=True)
    
    return results_df,cost_threshold, high_cost_hours