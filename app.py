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
            st.session_state['base_incenitve_DF'] = float(get_param('base_incenitve_DF'))
            st.session_state['base_incenitve_DR'] = float(get_param('base_incenitve_DR'))
            st.session_state['battery_energy_capacity'] = float(get_param('battery_energy_capacity'))
            st.session_state['clusters_rqd'] = ast.literal_eval(get_param('clusters_rqd'))
            st.session_state['num_slots'] = int(len(st.session_state['demand']))
            st.session_state['mode'] = ast.literal_eval(get_param('mode'))
            st.session_state['inconvenience_cost'] = ast.literal_eval(get_param('inconvenience_cost'))
            st.session_state['output_folder_name'] = get_param('Output Folder Name')
            st.session_state['solar_pu_cost'] = float(get_param('solar_pu_cost'))
            st.session_state['wind_pu_cost'] = float(get_param('wind_pu_cost'))
            st.session_state['market_limit'] = float(get_param('market_limit'))
            st.session_state['battery_power_capacity'] = float(get_param('battery_power_capacity'))
            st.session_state['battery_initial_state'] = float(get_param('battery_initial_state'))
            st.session_state['Re_forecast'] = ast.literal_eval(get_param('RE_forecasting'))
            st.session_state['Daily_slots'] = int(float(get_param('daily_slots')))
            
            st.sidebar.success("Data successfully initialized!")

        except Exception as e:
            st.sidebar.error(f"Error parsing files: {e}")

# --- 3. Using Data for Analysis/Plotting ---
st.title("Demand Flexibility Analysis Dashboard")

c1,c2=st.columns([1,2])

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

    # Use a spinner for visual feedback
        with st.spinner("Running optimization model..."):
            print(len(st.session_state.flex))
            progress_bar = st.progress(0)
            status_text = st.empty()

        
            i=0
            for flex in st.session_state.flex:
                progress_bar.progress(i / len(st.session_state.flex))
                status_text.text(f"Processing flexibility level {flex} of {(st.session_state.flex)}")
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
                st.session_state.availability
            )
            
            # Store results in session state DataFrames
                st.session_state["pu_gen_cost_flex"][flex] = pu_gen_cost
                st.session_state["total_cost_flex"][flex] = total_cost
                st.session_state["total_cost_sum"].loc[flex] = total_cost.sum()
                st.session_state["opt_demand"][flex] = opt_demand
                st.session_state["Battery"][flex]=battery_profile
                st.session_state["market_power"][flex]=market_power
                st.session_state["schedule_gen"][flex]=schedule_gen.sum(axis=1)
            
            # Update progress bar and status text
                progress_bar.progress((i + 1) / len(st.session_state.flex))
            # status_text.text(f"Processing flexibility level {flex} of {(st.session_state.flex)}")
                i=i+1
        

                
        st.success("Analysis Completed successfully!")
    
        st.rerun() 

    if st.session_state.get("total_cost_flex") is not None:
        status_text = st.empty()
        status_text.text(st.success("Analysis Completed successfully!"))

          


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
                total_cost = st.session_state["total_cost_sum"].loc[flex, 'total_cost_sum']
                total_cost_saving = st.session_state["total_cost_sum"].loc[0, 'total_cost_sum'] - total_cost
                percent_saving = (total_cost_saving / st.session_state["total_cost_sum"].loc[0, 'total_cost_sum']) * 100
    
                data.loc[i] = [flex, total_cost, total_cost_saving, percent_saving]

            
      
            st.dataframe(data,hide_index=True,height=300,use_container_width=True)
      

        
        
    
st.divider()
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
if "DF_pu_savings" in st.session_state:
    dd1, dd2 = st.columns([1,2])

    with dd1:
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
        if "opt_demand" in st.session_state:
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
                savings_matrix = pd.DataFrame(columns=range(365), index=range(daily_slots))
                data = st.session_state["DF_pu_savings"][selected_flex]
                # data =st.session_state['pu_gen_cost_flex'][0].values - st.session_state['pu_gen_cost_flex'][selected_flex].values
                # data =st.session_state['pu_gen_cost_flex'][selected_flex].values
                for i in range(365):
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

dd3, dd4 = st.columns([1,2])
month_dates = month_info(2024)

from iex_data import scrape_date_range
import datetime as dt



with dd3:
        available_month_options = list(month_dates.keys())
        selected_month = st.selectbox(
        "Select month:",
        options=available_month_options) 

cc1, cc2 = st.columns([6,1])
with cc1:
        df1=st.session_state["opt_demand"][selected_flex]
        df0=st.session_state["opt_demand"][0]
        daily_slots = st.session_state.get("Daily_slots", 24)  # Default to 24 if not set

        # Create cost matrix
        cost_matrix = pd.DataFrame(columns=range(365), index=range(daily_slots))
        data = df1.values - df0.values
        for i in range(365):
                start_idx = i * daily_slots
                end_idx = (i + 1) * daily_slots

                if end_idx <= len(data):
                    cost_matrix.iloc[:, i] = data[start_idx:end_idx]


        month_data = month_dates[selected_month]
        month_cost_data = cost_matrix.iloc[:, month_data['start_day']-1:month_data['end_day']]

        fig11 = px.box(np.array(month_cost_data).T,
               title='Demand Modulation Distribution by Hour')

           # Customize layout
        fig11.update_layout(
                xaxis_title="Hour of Day",
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

        st.plotly_chart(fig11, use_container_width=True)




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

with d1:
    

    st.subheader("DAM MCP trend for last 7 days")
    
    mcpdf = (st.session_state["mcp-dam"])
    print(mcpdf)
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

with d2:


    st.subheader("RTM MCP trend for last 7 days")
    
    mcpdf = (st.session_state["mcp-rtm"])
    print(mcpdf)
    fig = go.Figure() 
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
st.subheader("Consumer data analysis")
                
file_path = st.text_input(
    "Enter File Path for meter data:",
    placeholder="C:/Users/YourName/data/file.csv",
    help="Enter the full path to your file with meter data in CSV format"
)

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
[st.session_state["Consumer_data_results"],st.session_state['cost_threshold'], st.session_state['cost_hours']]= analyze_high_cost_contributors(st.session_state["consumer_data"], pd.DataFrame(st.session_state["pu_gen_cost_flex"][selected_flex]), float(high_cost_percentile) * 100)


st.divider()
st.subheader("High cost period analysis")
st.write(f"Cost threshold: {st.session_state['cost_threshold']} INR/kWh")

st.write(f"Number of high-cost hours: {st.session_state['cost_hours']} out of {len(st.session_state['pu_gen_cost_flex'][selected_flex])} total hours")
st.dataframe(st.session_state["Consumer_data_results"], use_container_width=True)   

m1,m2 = st.columns([1,1])
with m1:
        available_meters_options = list(st.session_state["consumer_data"].columns)
        selected_meter = st.selectbox(
        "Consumer meter:",
        options=available_meters_options) 

mg1,mg2 = st.columns([1,1])
with mg1:
    st.subheader("Selected consumer meter data")
    meter_data = st.session_state["consumer_data"][selected_meter]
    # st.table(meter_data)
    fig = go.Figure(data=go.Scatter(
        x=meter_data,
        y=st.session_state["pu_gen_cost_flex"][selected_flex],
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


