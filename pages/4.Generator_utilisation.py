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


st.title("Generation scheduling and cost overview")

m1,m2 = st.columns([1,3])
with m1:
        if st.session_state.get("generators_schedule") is not None:
            available_flex_options = list(st.session_state["opt_demand"].columns)
            selected_flex = st.selectbox(
            "Flexibility:",
            options=available_flex_options) 
               
print(st.session_state["ppa"]['Plant'].unique())
if selected_flex is not None:
     gen_details = pd.DataFrame(index=range(st.session_state["ppa"]['Plant'].count()), columns=['Plant','Capacity (MW)','Load Factor (%)','Maximum generation (MW)','Minimum generation (MW)'])
     gen_details['Plant'] = st.session_state["ppa"]['Plant']
     gen_flex = st.session_state["generators_schedule"][selected_flex]
     gen_flex.columns = st.session_state["ppa"]['Plant']
     for plant in st.session_state["ppa"]['Plant'].unique():
         
        gen_details.loc[gen_details['Plant'] == plant,'Capacity (MW)'] = round(st.session_state["ppa"].loc[st.session_state["ppa"]['Plant'] == plant, 'Capacity'].values[0])
        gen_details.loc[gen_details['Plant'] == plant,'Maximum generation (MW)'] = round(gen_flex[plant].max())
        gen_details.loc[gen_details['Plant'] == plant,'Minimum generation (MW)'] = round(gen_flex[plant].min())
        gen_details.loc[gen_details['Plant'] == plant,'Load Factor (%)'] = round((gen_flex[plant].sum() / (st.session_state.num_slots * st.session_state["ppa"].loc[st.session_state["ppa"]['Plant'] == plant, 'Capacity'].values[0])) * 100, 2)
     
     st.subheader(f"Generation details for {selected_flex}")
     st.dataframe(gen_details, hide_index=True, use_container_width=True) 

#contribution of generator to demand
if selected_flex is not None:
   matrix = np.array(st.session_state["generators_schedule"][selected_flex])
   vector = np.array(st.session_state["opt_demand"][selected_flex])
   st.session_state["generator_contribution"] = pd.DataFrame(matrix / vector[:, None], columns=st.session_state["generators_schedule"][selected_flex].columns)

   m1,m2 = st.columns([1,3])
   with m1:
         if st.session_state.get("generators_schedule") is not None:
               available_gen_options = list(st.session_state["generators_schedule"][selected_flex].columns)
               selected_gen = st.selectbox(
               "Generator:",
               options=available_gen_options) 

   m1,m2 = st.columns([1,2])
   
   with m1:
      if selected_gen is not None:
         st.subheader(f"Contribution of {selected_gen} to demand for {selected_flex*100} % flexibility")
         st.write(f"Variable cost: {round(st.session_state['ppa'].loc[st.session_state['ppa']['Plant'] == selected_gen, 'Variable cost'].values[0], 2)} INR/kWh")
         st.write(f"Installed capacity: {round(st.session_state['ppa'].loc[st.session_state['ppa']['Plant'] == selected_gen, 'Capacity'].values[0], 2)} MW")
         st.write(f"Type: {st.session_state['ppa'].loc[st.session_state['ppa']['Plant'] == selected_gen, 'Type'].values[0]}")
   with m2:
      
      if selected_flex and selected_gen is not None:
         num_slots = st.session_state["generator_contribution"].shape[0]
         daily_slots = st.session_state['Daily_slots']
         savings_matrix= pd.DataFrame(columns=range(num_slots//daily_slots), index=range(daily_slots))
         data = 100*st.session_state["generator_contribution"][selected_gen]

         for i in range(num_slots//daily_slots):
            start_idx = i * daily_slots
            end_idx = (i + 1) * daily_slots

            if end_idx <= len(data):
               savings_matrix.iloc[:, i] = data[start_idx:end_idx]



         fig12 = go.Figure(data=go.Heatmap(
         z=savings_matrix.T.values,
         colorscale='Viridis',  # You can change this: 'Plasma', 'Inferno', 'Hot', etc.
         colorbar=dict(title="(%)")
         ))


         fig12.update_layout(
         
         xaxis_title='Time Slots (Hours)',
         yaxis_title='Day of Year',
         height=500,
         width=900,
         template='plotly_white'
         )

      # Display in Streamlit
         st.plotly_chart(fig12, use_container_width=True)