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
# import generation_cost2

import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.title("Consumer demand analysis for high-cost periods")

                
file_path = st.text_input(
    "Enter File Path for meter data:",
    placeholder="C:/Users/YourName/data/file.csv",
    help="Enter the full path to your file with meter data in CSV format"
)
# st.session_state["consumer_data"] = None
from  analyze_high_cost_contributors import analyze_high_cost_contributors
if button := st.button("Load Meter Data"):
    
    if file_path is not None and file_path.endswith('.csv'):
        st.session_state["consumer_data"]= pd.read_csv(file_path)
        st.success("Meter data loaded successfully!")
    else:
        st.warning("Please enter a valid file path ending with .csv")
    

c1,c2 = st.columns([1,4])
with c1:
     high_cost_percentile = st.number_input("Enter the thershold percentile for PU gen cost:", key="High cost thershold percentile", placeholder="0.1", help="Enter the percentile level for which you want to analyze consumer demand convergence", min_value=0.0, max_value=100.0, value=10.0)
if "consumer_data" in st.session_state and "pu_gen_cost_flex" in st.session_state :
    [st.session_state["Consumer_data_results"],st.session_state['cost_threshold'], st.session_state['cost_hours']]= analyze_high_cost_contributors(st.session_state["consumer_data"], pd.DataFrame(st.session_state["pu_gen_cost_flex"][0]), float(high_cost_percentile) )


st.divider()
st.subheader("High cost period analysis")
if 'cost_threshold' in st.session_state and 'cost_hours' in st.session_state:
    st.write(f"Cost threshold: {round(st.session_state['cost_threshold'], 2)} INR/kWh")

    st.write(f"Number of high-cost hours: {st.session_state['cost_hours']} out of {len(st.session_state['pu_gen_cost_flex'][0])} total hours")
    st.dataframe(st.session_state["Consumer_data_results"], use_container_width=True)   

m1,m2 = st.columns([1,1])
with m1:
        if st.session_state.get("consumer_data") is not None:
            available_meters_options = list(st.session_state["consumer_data"].columns)
            selected_meter = st.selectbox(
            "Consumer meter:",
            options=available_meters_options) 

mg1,mg2 = st.columns([1,1])


if "pu_gen_cost_flex" in st.session_state and  "consumer_data" in st.session_state and selected_meter in st.session_state["consumer_data"].columns:
    with mg1:
        
        st.subheader("Selected consumer meter data")
        meter_data = st.session_state["consumer_data"][selected_meter]
        # st.table(meter_data)
        fig = go.Figure(data=go.Scatter(
            x=meter_data,
            y=st.session_state["pu_gen_cost_flex"][0],
            mode='markers',     
        ))

        fig.update_layout(
            title='Correlation: Consumer load vs PU generation cost',
            xaxis_title='Consumer demand spread (kW)',
            yaxis_title='PU gen cost (INR/kWh)',
            height=500,
            width=900,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)


    with mg2:
        st.subheader("Meter data heatmap")



        df1=st.session_state["consumer_data"][selected_meter]
        daily_slots = st.session_state.get("Daily_slots", 24)  # Default to 24 if not set

        # Create cost matrix
        cost_matrix = pd.DataFrame(columns=range(365), index=range(daily_slots))
        data = df1.values
        for i in range(365):
                start_idx = i * daily_slots
                end_idx = (i + 1) * daily_slots

                if end_idx <= len(data):
                    cost_matrix.iloc[:, i] = data[start_idx:end_idx]


        fig = go.Figure(data=go.Heatmap(
            z=cost_matrix.T.values,
            colorscale='Viridis',  # You can change this: 'Plasma', 'Inferno', 'Hot', etc.
            colorbar=dict(title="KW")
            ))


        fig.update_layout(
            title='Heat Map: Consumer load',
            xaxis_title='Time Slots (Hours)',
            yaxis_title='Day of Year',
            height=500,
            width=900,
            template='plotly_white'
        )

        st.session_state['total_cost_flex'].to_excel(r"C:\Users\Kaustubh\OneDrive\Desktop\total_cost.xlsx")
        st.session_state['opt_demand'].to_excel(r"C:\Users\Kaustubh\OneDrive\Desktop\opt_demnad.xlsx")


    # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

