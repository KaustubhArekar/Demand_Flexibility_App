import streamlit as st

from code import interact

from matplotlib import widgets
import scipy as sp
import streamlit as st
import pandas as pd
import ast
# from datetime import datetime, timedelta

from tomlkit import datetime
import analyze_high_cost_contributors
from generation_cost import generation_cost
from month_info import month_info
import plotly.express as px

import matplotlib.pyplot as plt
import numpy as np  
import generation_cost2

import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.title("Charts and Analysis")

# Check if data exists from page 1
if st.session_state.get("pu_gen_cost_flex") is None:
    st.warning("⚠️ Please run optimization on the main page first")
    st.stop()  # Stop execution



num_slots = st.session_state.get("num_slots")

c11,c12=st.columns([1,1])
with c11:
     if st.button("Calculate parameters for charts", use_container_width=True):
          with st.spinner("Calculating parameters..."):
               st.session_state["DF_pu_savings"] = pd.DataFrame(index=range(st.session_state.num_slots), columns=st.session_state.flex)
               st.session_state["Demand_modulation"] = pd.DataFrame(index=range(st.session_state.num_slots), columns=st.session_state.flex)
               for i in range(st.session_state.num_slots):
                    for flex in range(len(st.session_state.flex)):
                        st.session_state["Demand_modulation"].iloc[i,flex] = st.session_state["opt_demand"].iloc[i, 0] - st.session_state["opt_demand"].iloc[i, flex]
                        if st.session_state["Demand_modulation"].iloc[i, flex]> 0.01*st.session_state["opt_demand"].iloc[i, 0]:
                            st.session_state["DF_pu_savings"].iloc[i, flex] = (st.session_state["total_cost_flex"].iloc[i, 0] - st.session_state["total_cost_flex"].iloc[i, flex])/st.session_state["Demand_modulation"].iloc[i, flex]
                        elif st.session_state["Demand_modulation"].iloc[i, flex] < -0.01*st.session_state["opt_demand"].iloc[i, 0]:
                            st.session_state["DF_pu_savings"].iloc[i, flex] = -(st.session_state["total_cost_flex"].iloc[i, 0] - st.session_state["total_cost_flex"].iloc[i, flex])/st.session_state["Demand_modulation"].iloc[i, flex]
                        else:
                            st.session_state["DF_pu_savings"].iloc[i, flex] = 0

          st.rerun()

# col1, col2 = st.tabs(["Optimized Demand Comparison", "Per unit savings from DF/DR"])
selected_flex =0
selected_month ='january'
if "DF_pu_savings" in st.session_state:
    dd1, dd2 = st.columns([1,2])

    with dd1:
            if st.session_state.get("opt_demand") is not None:
                available_flex_options = list(st.session_state["opt_demand"].columns)
                selected_flex = st.selectbox(
                "Select Flexibility Level:",
                options=available_flex_options)

    col1, col2 = st.columns([1,1])

        
    with col1:
                if "opt_demand" in st.session_state and selected_flex in st.session_state["opt_demand"].columns:
                    df1 = st.session_state["opt_demand"][selected_flex]
                    df2=st.session_state["Battery"][selected_flex]
                    df3=st.session_state["market_power"][selected_flex]
                    df4=st.session_state["schedule_gen"][selected_flex]   
                    df5=st.session_state["re"].sum(axis=1) 
                
                # Initialize the Plotly Figure
                    fig = go.Figure()
                
                # Add a trace for every column in the dataframe
                # This automatically handles multiple flexibility levels (e.g., 0, 0.1, 0.2)
                    
                    fig.add_trace(go.Scatter(
                    x=df1.index, 
                    y=df1,
                    mode='markers',
                    name=f'Optimized Demand (Flex={selected_flex})',
                    marker=dict(size=5, symbol='x', color='white'),
                    # stackgroup='Two', 
                    hovertemplate="Time: %{x}<br>Optimised demand: %{y:.2f} MW<extra></extra>"))

                    fig.add_trace(go.Scatter(
                    x=df4.index, 
                    y=df4,
                    mode='lines',
                    name=f'Schedule Generation (Flex={selected_flex})',
                    stackgroup='one', 
                    hovertemplate="Time: %{x}<br>Schedulled generation: %{y:.2f} Mw<extra></extra>"))

                    fig.add_trace(go.Scatter(
                    x=df5.index, 
                    y=df5,
                    mode='lines',
                    name=f'Renewable Energy (Flex={selected_flex})',
                    stackgroup='one', 
                    hovertemplate="Time: %{x}<br>RE: %{y:.2f} MW<extra></extra>"))

                    fig.add_trace(go.Scatter(
                    x=df2.index, 
                    y=-df2,
                    mode='lines',
                    name=f'Battery power (Flex={selected_flex})',
                    stackgroup='one', 
                    hovertemplate="Time: %{x}<br>Battery profile: %{y:.2f} MW<extra></extra>"))

                    fig.add_trace(go.Scatter(
                    x=df3.index, 
                    y=df3,
                    mode='lines',
                    name=f'Market Power (Flex={selected_flex})',
                    stackgroup='one', 
                    hovertemplate="Time: %{x}<br>Market Power: %{y:.2f} MW<extra></extra>"))
                    
                    
                
                # Update Layout for better UX
                    fig.update_layout(
                    title="Demand, Schedulled power, RE, Market power and Battery profile",
                    xaxis_title="Time Steps",
                    yaxis_title="MW",
                    hovermode="x unified",  # Shows all values in one tooltip
                    legend=dict(
                    # font=dict(size=8),
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5
                    ),
                    margin=dict(b=100),
                    template="plotly_white"
                )
                
                # Display in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data found for Optimized Demand Comparison. Please run the analysis first.")
        
    with col2:


            #     st.subheader("Heat Map: Slot-wise Per Unit Generation Cost")
                df1=st.session_state["opt_demand"][selected_flex]
                df0=st.session_state["opt_demand"][0]
                daily_slots = st.session_state.get("Daily_slots", 24)  # Default to 24 if not set
            
            # Create cost matrix
                cost_matrix = pd.DataFrame(columns=range(num_slots//daily_slots), index=range(daily_slots))
                data = df1.values
                for i in range(num_slots//daily_slots):
                    start_idx = i * daily_slots
                    end_idx = (i + 1) * daily_slots
            
                    if end_idx <= len(data):
                        cost_matrix.iloc[:, i] = data[start_idx:end_idx]
            
        
                fig = go.Figure(data=go.Heatmap(
                z=cost_matrix.T.values,
                colorscale='Viridis',  # You can change this: 'Plasma', 'Inferno', 'Hot', etc.
                colorbar=dict(title="MW")
                ))
            
            
                fig.update_layout(
                title='Heat Map: Demand',
                xaxis_title='Time Slots (Hours)',
                yaxis_title='Day of Year',
                height=500,
                width=900,
                template='plotly_white'
                )
            
            # Display in Streamlit
                st.plotly_chart(fig, use_container_width=True)

    # Monthly analysis and visualisation 
    #1. DF/DR needed hourly distribution (Box plot for each month)
    dfcs1, dfcs2 = st.columns([1,1])
    with dfcs1:
        if "DF_pu_savings" in st.session_state:
                    df1 = st.session_state["opt_demand"][selected_flex]
                    df2=st.session_state["opt_demand"][0]
                    
                # Initialize the Plotly Figure
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                    x=df1.index, 
                    y=df1,
                    mode='lines',
                    name=f'Optimized Demand (Flex={selected_flex})',
                    
                    # stackgroup='Two', 
                    hovertemplate="Time: %{x}<br>Optimised demand: %{y:.2f} MW<extra></extra>"))

                    fig.add_trace(go.Scatter(
                    x=df2.index, 
                    y=df2,
                    mode='lines',
                    name=f'Base demand',
                    
                    hovertemplate="Time: %{x}<br>Base demand: %{y:.2f} MW<extra></extra>"))

                    fig.update_layout(
                    title="Base demand and optimised demand comparison",
                    xaxis_title="Time Steps",
                    yaxis_title="MW",
                    hovermode="x unified",  # Shows all values in one tooltip
                    legend=dict(
                    # font=dict(size=8),
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5
                    ),
                    margin=dict(b=100),
                    template="plotly_white")

                    st.plotly_chart(fig, use_container_width=True)
                
    with dfcs2:
                savings_matrix = pd.DataFrame(columns=range(num_slots//daily_slots), index=range(daily_slots))
                data = (10**3)*st.session_state["DF_pu_savings"][selected_flex]

                # data =st.session_state['pu_gen_cost_flex'][0].values - st.session_state['pu_gen_cost_flex'][selected_flex].values
                # data =st.session_state['pu_gen_cost_flex'][selected_flex].values
                for i in range(num_slots//daily_slots):
                    start_idx = i * daily_slots
                    end_idx = (i + 1) * daily_slots

                    if end_idx <= len(data):
                        savings_matrix.iloc[:, i] = data[start_idx:end_idx]



                fig12 = go.Figure(data=go.Heatmap(
                z=savings_matrix.T.values,
                colorscale='Viridis',  # You can change this: 'Plasma', 'Inferno', 'Hot', etc.
                colorbar=dict(title="(INR/kWh)")
                ))
            
            
                fig12.update_layout(
                title='Heat Map: Change in generation cost per unit DF/DR',
                xaxis_title='Time Slots (Hours)',
                yaxis_title='Day of Year',
                height=500,
                width=900,
                template='plotly_white'
                )

                # fig.update_layout(
                #    title='Heat Map: DF/DR activated',
                #    xaxis_title='Time Slots (Hours)',
                #    yaxis_title='Day of Year',
                #    height=500,
                #    width=900,
                #    template='plotly_white'
                # )
            
            # Display in Streamlit
                st.plotly_chart(fig12, use_container_width=True)
