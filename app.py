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

st.set_page_config(layout="wide")

# --- 1. Sidebar for File Uploads ---
with st.sidebar:
    st.title("Data Input")
    st.info("Upload the required files below.")
    
    file_configs = {
        "Generation Stack": "Generation_stack.xlsx",
        "Demand Data": "demand.csv",
        "Market Rates": "market_rate.xlsx",
        "Renewable Energy": "RE.xlsx",
        "Input Parameters": "input_parameters.csv",
        "Generator availability": "availability.xlsx"
    }

    uploaded_data = {}
    for label, filename in file_configs.items():
        uploaded_data[label] = st.file_uploader(f"Upload {filename}", type=['csv', 'xlsx'], key=label)

# --- 2. Data Processing Logic ---
# Check if all files are uploaded
if all(uploaded_data.values()):
    if st.sidebar.button("⚙️ Process & Initialize Data"):
        try:
            # Read Dataframes from memory
            ppa = pd.read_excel(uploaded_data["Generation Stack"])
            projected_demand = pd.read_csv(uploaded_data["Demand Data"])
            market_price = pd.read_excel(uploaded_data["Market Rates"])
            re = pd.read_excel(uploaded_data["Renewable Energy"])
            input_parameters = pd.read_csv(uploaded_data["Input Parameters"])
            availability= pd.read_excel(uploaded_data["Generator availability"])

            # Helper to extract values from the parameters dataframe
            def get_param(p_name):
                return input_parameters[input_parameters['Parameter'] == p_name]['Value'].values[0]

            # Store processed variables in st.session_state
            st.session_state['ppa'] = ppa
            st.session_state['demand'] = projected_demand
            st.session_state['market'] = market_price
            st.session_state['re'] = re
            st.session_state['availability'] = availability

            # Parsing specific variables
            st.session_state['flex'] = ast.literal_eval(get_param('flexibility'))
            # st.session_state['base_incenitve_DF'] = float(get_param('base_incenitve_DF'))
            # st.session_state['base_incenitve_DR'] = float(get_param('base_incenitve_DR'))
            st.session_state['battery_energy_capacity'] = float(get_param('battery_energy_capacity'))
            # st.session_state['clusters_rqd'] = ast.literal_eval(get_param('clusters_rqd'))
            st.session_state['num_slots'] = int(len(st.session_state['demand']))
            # st.session_state['mode'] = ast.literal_eval(get_param('mode'))
            # st.session_state['inconvenience_cost'] = ast.literal_eval(get_param('inconvenience_cost'))
            st.session_state['output_folder_name'] = get_param('Output Folder Name')
            st.session_state['solar_pu_cost'] = float(get_param('solar_pu_cost'))
            st.session_state['wind_pu_cost'] = float(get_param('wind_pu_cost'))
            st.session_state['market_limit'] = float(get_param('market_limit'))
            st.session_state['battery_power_capacity'] = float(get_param('battery_power_capacity'))
            st.session_state['battery_initial_state'] = float(get_param('battery_initial_state'))
            st.session_state['Re_forecast'] = ast.literal_eval(get_param('RE_forecasting'))
            st.session_state['Daily_slots'] = int(float(get_param('daily_slots')))
            st.session_state['max_daily_shift'] = float(get_param('limit_on_totalshift'))

            st.sidebar.success("Data successfully initialized!")

        except Exception as e:
            st.sidebar.error(f"Error parsing files: {e}")

# --- 3. Using Data for Analysis/Plotting ---
st.title("Demand Flexibility Analysis Model")
st.write("The Demand Flexibility Analysis Model is a specialized analytical tool designed to evaluate the economic and operational impact of demand-side management on power systems by processing multi-faceted data inputs—including Power Purchase Agreements (PPA), projected demand, renewable energy availability, and market rates—the underlying model calculates the total generation cost across various flexibility levels. The dashboard enables users to visualize demand modulation and determine per-unit savings derived from Demand Flexibility (DF) and Demand Response (DR) initiatives. Furthermore, it integrates external IEX market data, such as Day-Ahead Market (DAM) and Real-Time Market (RTM) trends, to benchmark internal costs against market prices. For granular insights, the model identifies high-cost periods and analyzes consumer meter data to pinpoint specific contributors to peak generation costs, helping stakeholders optimize demand patterns for maximum financial efficiency.")
c1,c2=st.columns([2,3])

with c1:
    

    if st.button("Start Analysis", use_container_width=True):
    # Ensure result placeholders exist in session state
        st.session_state["pu_gen_cost_flex"] = pd.DataFrame(index=range(st.session_state.num_slots))
    
    
        st.session_state["opt_demand"] = pd.DataFrame(index=range(st.session_state.num_slots))
        st.session_state["total_cost_flex"] = pd.DataFrame(index=range(st.session_state.num_slots))
        st.session_state["total_cost_sum"] = pd.DataFrame(index=st.session_state.flex, columns=['total_cost_sum'])
        st.session_state["schedule_gen"] = pd.DataFrame(index=range(st.session_state.num_slots))
        st.session_state["market_power"] = pd.DataFrame(index=range(st.session_state.num_slots))
        st.session_state["Battery"] = pd.DataFrame(index=range(st.session_state.num_slots))
        st.session_state["generators_schedule"] = {}

    # Use a spinner for visual feedback
        with st.spinner("Running optimization model..."):
            # print(len(st.session_state.flex))
            progress_bar = st.progress(0)
            status_text = st.empty()

        
            i=0
            for flex in st.session_state.flex:
                progress_bar.progress(i / len(st.session_state.flex))
                # st.altair_chart(st.altair_chart(go.Scatter(x=[i], y=[flex], mode='markers')))
            # Note: passing 'flex' (the current value) instead of 'st.session_state.flex' (the list)
                [total_cost, pu_gen_cost, demand, opt_demand, net_demand, schedule_gen, market_power, battery_profile] = generation_cost2.generation_cost2(
                flex, 
                st.session_state.demand, 
                st.session_state.re, 
                st.session_state.market, 
                st.session_state.ppa, 
                st.session_state.battery_energy_capacity, 
                st.session_state.battery_power_capacity, 
                st.session_state.battery_initial_state, 
                st.session_state.solar_pu_cost, 
                st.session_state.wind_pu_cost,
                st.session_state.market_limit,
                st.session_state.output_folder_name,
                st.session_state.num_slots,
                st.session_state.Daily_slots,
                st.session_state.availability,
                st.session_state.max_daily_shift
            )
                print(schedule_gen)
            # Store results in session state DataFrames
                st.session_state["pu_gen_cost_flex"][flex] = pu_gen_cost
                st.session_state["total_cost_flex"][flex] = total_cost
                st.session_state["total_cost_sum"].loc[flex] = total_cost.sum()
                st.session_state["opt_demand"][flex] = opt_demand
                st.session_state["Battery"][flex]=battery_profile
                st.session_state["market_power"][flex]=market_power
                st.session_state["schedule_gen"][flex]=schedule_gen.sum(axis=1)
                st.session_state["generators_schedule"][flex]=schedule_gen
            
            # Update progress bar and status text
                progress_bar.progress((i + 1) / len(st.session_state.flex))
            # status_text.text(f"Processing flexibility level {flex} of {(st.session_state.flex)}")
                i=i+1
        

        print(st.session_state["generators_schedule"][flex] )       
        st.success("Analysis Completed successfully!")
    
        st.rerun() 

    if st.session_state.get("total_cost_flex") is not None:
        # status_text = st.empty()
        # status_text.text(st.success("Analysis Completed successfully!"))
        st.success("Analysis Completed successfully!")

          


c1,c2=st.columns([1,1])           



with c1:
    if "total_cost_sum" in st.session_state:
        # Create figure
        fig0 = go.Figure()
        flex_array = np.array(st.session_state["flex"]).flatten()
        # Add bar trace
        fig0.add_trace(go.Bar(
            x=[f"{x*100:.0f}%" for x in flex_array],  # Flexibility levels on x-axis
            y=np.array(st.session_state['total_cost_sum'].values).flatten(),  # Total cost sums on y-axis
            name='Total Cost',
            marker_color='indianred',
            # text=np.array(st.session_state.total_cost_sum).round(0),
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate="<b>Flexibility: %{x}%</b><br>" +
                        "Total Cost: ₹%{y:,.0f}<br>" +
                        "<extra></extra>"
        ))
        
        # Update layout
        fig0.update_layout(
             
            yaxis=dict(range=[min(np.array(st.session_state['total_cost_sum'].values).flatten())*0.95, max(np.array(st.session_state['total_cost_sum'].values).flatten())*1.005]), 
            title={
                'text': "Total Cost of Generation for Different Flexibility Levels",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title={
                'text': "Flexibility Level (%)",
                'font': {'size': 12}
            },
            yaxis_title={
                'text': "Total Cost (₹)",
                'font': {'size': 12}
            },
            hovermode="x unified",
            height=400,
            template="plotly_white",
            showlegend=False,  # Since only one trace, legend is redundant
            bargap=0.3  # Add gap between bars
        )
        
        # Optional: Add grid lines for better readability
        # fig0.update_yaxis(gridcolor='lightgray', gridwidth=0.5)
        
        # Display in Streamlit
        st.plotly_chart(fig0, use_container_width=True)
        
    else:
        st.info("No cost data available. Please run the analysis first.")

        

with c2:
        if "total_cost_sum" in st.session_state:
            st.subheader("Summary")
            data = pd.DataFrame(index = range(len(st.session_state.flex)), columns=['Flexibility','Total Cost (Million INR)', 'Total Cost Savings (Million INR)', '% Cost Savings'])
            for i, flex in enumerate(st.session_state.flex):
                total_cost = round(st.session_state["total_cost_sum"].loc[flex, 'total_cost_sum'],0)
                total_cost_saving = round(st.session_state["total_cost_sum"].loc[0, 'total_cost_sum'] - total_cost,0)
                percent_saving = (total_cost_saving / st.session_state["total_cost_sum"].loc[0, 'total_cost_sum']) * 100
    
                data.loc[i] = [flex, total_cost, total_cost_saving, percent_saving]

            
      
            st.dataframe(data,hide_index=True,use_container_width=True)
      

        
        
    
