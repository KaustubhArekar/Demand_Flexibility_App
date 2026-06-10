def read_data():
    
    import ast
    import pandas as pd

    INPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/input data 2/'
    OUTPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/output data/'

    # INPUT_PATH = input("Provide path for input data")
    # OUTPUT_PATH = input("Provide path for output folder")
    # path_for_meter_data = input("Provide path for meter data")


    ppa =pd.read_excel(INPUT_PATH + 'Generation_stack.xlsx') # PPA data of thermal plants (Capacity, ramping limts, technical min, variable cost)
    print('Succuefully read: Generation_stack.xlsx')
    projected_demand = pd.read_csv(INPUT_PATH + 'demand.csv') # Discom hourly demand in MW 
    print('Succuefully read: demand.csv')
    market_price = pd.read_excel(INPUT_PATH+'market_rate.xlsx') # Power market RTM prices for last year
    print('Succuefully read: market_rate.xlsx')
    re = pd.read_excel(INPUT_PATH +'RE - high.xlsx') 
    print('Succuefully read: RE - high.xlsx')
    # tariff = pd.read_csv(INPUT_PATH +'tariff.csv') # cluster wise hourly tariff
    # print('Succuefully read: tariff.csv')
    input_parameters = pd.read_csv(INPUT_PATH +'input_parameters.csv') # User defined parameters
    print('Succuefully read: input_parameters.csv')

    flex =ast.literal_eval(input_parameters[input_parameters['Parameter']=='flexibility']['Value'].values[0])
    base_incenitve_DF =float(input_parameters[input_parameters['Parameter']=='base_incenitve_DF']['Value'].values[0])
    base_incenitve_DR = float(input_parameters[input_parameters['Parameter']=='base_incenitve_DR']['Value'].values[0])
    DR_lambda = float(input_parameters[input_parameters['Parameter']=='DR_lambda']['Value'].values[0])
    DF_lambda = float(input_parameters[input_parameters['Parameter']=='DF_lambda']['Value'].values[0])
    daily_slots = int(float(input_parameters[input_parameters['Parameter']=='daily_slots']['Value'].values[0]))
    step_size = float(input_parameters[input_parameters['Parameter']=='step_size']['Value'].values[0])
    solar_pu_cost =float(input_parameters[input_parameters['Parameter']=='solar_pu_cost']['Value'].values[0])
    wind_pu_cost =float(input_parameters[input_parameters['Parameter']=='wind_pu_cost']['Value'].values[0])
    market_limit = float(input_parameters[input_parameters['Parameter']=='market_limit']['Value'].values[0])
    battery_power_capacity =float(input_parameters[input_parameters['Parameter']=='battery_power_capacity']['Value'].values[0])
    battery_energy_capacity =float(input_parameters[input_parameters['Parameter']=='battery_energy_capacity']['Value'].values[0])
    battery_initial_state = float(input_parameters[input_parameters['Parameter']=='battery_initial_state']['Value'].values[0])
    clusters_rqd = ast.literal_eval(input_parameters[input_parameters['Parameter']=='clusters_rqd']['Value'].values[0])
    mode=ast.literal_eval(input_parameters[input_parameters['Parameter']=='mode']['Value'].values[0])
    inconvenience_cost =ast.literal_eval(input_parameters[input_parameters['Parameter']=='inconvenience_cost']['Value'].values[0])
    output_folder_name =(input_parameters[input_parameters['Parameter']=='Output Folder Name']['Value'].values[0])
    Re_forecast =ast.literal_eval(input_parameters[input_parameters['Parameter']=='RE_forecasting']['Value'].values[0])

    return ppa, projected_demand, market_price, re, tariff, flex, base_incenitve_DF, base_incenitve_DR, DR_lambda, DF_lambda, daily_slots, step_size, solar_pu_cost, wind_pu_cost, market_limit, battery_power_capacity, battery_energy_capacity, battery_initial_state, clusters_rqd, mode,inconvenience_cost, Re_forecast, INPUT_PATH, OUTPUT_PATH, output_folder_name
