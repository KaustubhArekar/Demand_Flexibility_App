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

from iex_data import scrape_date_range
import datetime as dt

st.title("IEX Market MCP Data trend")
st.subheader("This feature works only on local machine. It will not work on Streamlit cloud due to IEX data access restrictions.")

if st.button("Load IEX market data", use_container_width=True):
    todays_date = dt.date.today()
    yesterday_date = todays_date - dt.timedelta(days=1)
    previous_week_date = todays_date - dt.timedelta(days=7)
    previous_week_date_1 = yesterday_date - dt.timedelta(days=7)
    st.session_state["mcp-dam"] = (scrape_date_range(previous_week_date, todays_date,"day-ahead-market"))
    st.session_state["mcp-rtm"] = (scrape_date_range(previous_week_date_1, yesterday_date,"real-time-market"))

    # st.success("IEX market data loaded successfully!")
    st.rerun()



d1,d2 = st.tabs(["DAM MCP TREND", "RTM MCP TREND"])

if "mcp-dam" in st.session_state:
    with d1:
        st.subheader("DAM MCP trend for last 7 days")
        mcpdf = (st.session_state["mcp-dam"])
        if mcpdf is not None and isinstance(mcpdf, pd.DataFrame) and 'Date' in mcpdf.columns:
            
                fig = go.Figure() 
                for date in mcpdf['Date'].unique():
                    daily_data = mcpdf.loc[mcpdf['Date'] == date]
                    # print(daily_data)
                    
                    fig.add_trace(go.Scatter(
                    x=daily_data.iloc[:, 2],  # Column 2 (index 1) - Time
                    y=daily_data.iloc[:, 7]/1000,  # Column 8 (index 7) - MCP
                    mode='lines',
                    name=f'{date}',
                    line=dict(width=2),
                    # marker=dict(size=4),
                    hovertemplate=f"Date: {date}<br>Time: %{{x}}<br>MCP: %{{y:.2f}} INR/MWh<extra></extra>"
                ))

                fig.update_layout(
                title="MCP Trend Comparison Across Dates",
                xaxis_title="Time Block",
                yaxis_title="MCP (INR/kWh)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template="plotly_white",
                height=500
                )
                st.plotly_chart(fig, use_container_width=True)

if "mcp-rtm" in st.session_state:
    with d2:

        st.subheader("RTM MCP trend for last 7 days")
        
        mcpdf = (st.session_state["mcp-rtm"])
        print(mcpdf)
        fig = go.Figure()
        if mcpdf is not None and isinstance(mcpdf, pd.DataFrame) and 'Date' in mcpdf.columns:
            for date in mcpdf['Date'].unique():
                daily_data = mcpdf.loc[mcpdf['Date'] == date]
                # print(daily_data)
                
                fig.add_trace(go.Scatter(
                x=daily_data.iloc[:, 3],  # Column 2 (index 1) - Time
                y=daily_data.iloc[:, 8]/1000,  # Column 8 (index 7) - MCP
                mode='lines',
                name=f'{date}',
                line=dict(width=2),
                # marker=dict(size=4),
                hovertemplate=f"Date: {date}<br>Time: %{{x}}<br>MCP: %{{y:.2f}} INR/MWh<extra></extra>"
            ))

            fig.update_layout(
            title="MCP Trend Comparison Across Dates",
            xaxis_title="Time Block",
            yaxis_title="MCP (INR/kWh)",
            hovermode="x unified",
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    

st.divider()
