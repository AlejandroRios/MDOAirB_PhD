"""
File name : Cruise performance function
Authors   : Alejandro Rios
Email     : aarc.88@gmail.com
Date      : November/2020
Last edit : November/2020
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil
Description:
    - This module calculates the cruise performance using the Breguet equations
Inputs:
    - Cruise altitude [ft]
    - Delta ISA [C deg]
    - Mach number
    - Mass at top of climb
    - Cruise distance [mn]
    - Vehicle dictionary
Outputs:
    - Cruise time [min]
    - Mass at top of descent [kg]
TODO's:
    - Rename variables 
"""
# =============================================================================
# IMPORTS
# =============================================================================
from inspect import isfunction
import numpy as np
# from scipy.optimize import fsolve
# from scipy.optimize import minimize
from scipy import optimize
from scipy.optimize import root
from framework.Performance.Engine.engine_performance import turbofan
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Attributes.Airspeed.airspeed import V_cas_to_mach, mach_to_V_cas, mach_to_V_tas, crossover_altitude
# from framework.Aerodynamics.aerodynamic_coefficients import zero_fidelity_drag_coefficient
from framework.Aerodynamics.aerodynamic_coefficients_ANN import aerodynamic_coefficients_ANN
from joblib import dump, load
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.80665
ft_to_m = 0.3048
km_to_nm = 0.539957
def payload_range(mass,mass_fuel, vehicle):

    
    aircraft = vehicle['aircraft']
    operations = vehicle['operations']
    wing = vehicle['wing']
    engine = vehicle['engine']

    if engine['type'] == 1:
        scaler_F = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
        nn_unit_F = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib')

        scaler_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
        nn_unit_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib')

    Sw = wing['area']
    W0 = mass

    Wf = mass_fuel

    delta_ISA = operations['flight_planning_delta_ISA']

    # altitude = operations['optimal_altitude_cruise']
    altitude = operations['cruise_altitude'] 

    mach = operations['mach_cruise']

    # Initialize product of all phases
    Mf = 1 - Wf/W0
    Mf_cruise = Mf
        
    ### Landing and Taxi
    Mf_cruise = Mf_cruise/0.992
        
    ### Loiter
    # Loiter at max L/D
    TSFC, L_over_D, fuel_flow, throttle_position = specific_fuel_consumption(
            vehicle, mach, altitude, delta_ISA, mass)
    #print(LDmax)

    TSFC = TSFC*1/3600

    # Factor to fuel comsumption
    # C_loiter = C_cruise*0.4/0.5
    C_loiter = TSFC*0.4/0.5
    #print(C_loiter)
    
    ### Cruise 2 (already known value)
    Mf_cruise = Mf_cruise/0.9669584006325017

    loiter_time = 60 * 45
    
    # Continue Loiter
    Mf_cruise = Mf_cruise/np.exp(-loiter_time*C_loiter/L_over_D)

    ### Descent
    Mf_cruise = Mf_cruise/0.99

    ### Start and warm-up
    Mf_cruise = Mf_cruise/0.99

    ### Taxi
    Mf_cruise = Mf_cruise/0.99

    ### Take-off
    Mf_cruise = Mf_cruise/0.995

    ### Climb
    Mf_cruise = Mf_cruise/0.98
    
    ### Cruise
    # Atmospheric conditions at cruise altitude
    # T,p,rho,mi = dt.atmosphere(altitude_cruise, 288.15)
    _, _, _, T, _, rho, _, a = atmosphere_ISA_deviation(altitude, delta_ISA)

    # Cruise speed
    V_tas = mach_to_V_tas(mach, altitude, delta_ISA)


    range_cruise = -(np.log(Mf_cruise)*V_tas*L_over_D)/TSFC



    return (range_cruise/1000)*km_to_nm


def specific_fuel_consumption(vehicle, mach, altitude, delta_ISA, mass):

    knots_to_meters_second = 0.514444

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    engine = vehicle['engine']

    if engine['type'] == 1:
        scaler_F = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
        nn_unit_F = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib')

        scaler_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
        nn_unit_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib')


    wing_surface = wing['area']

    V_tas = mach_to_V_tas(mach, altitude, delta_ISA)
    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(altitude, delta_ISA)

    CL_required = (2*mass*GRAVITY) / \
        (rho_ISA*((knots_to_meters_second*V_tas)**2)*wing_surface)
    # print('CL',CL_required)
    phase = 'cruise'
    # CD = zero_fidelity_drag_coefficient(aircraft_data, CL_required, phase)

    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 1
    CD_wing, _ = aerodynamic_coefficients_ANN(
        vehicle, altitude*ft_to_m, mach, CL_required, alpha_deg, switch_neural_network)

    friction_coefficient = 0.003
    CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

    CD = CD_wing + CD_ubrige

    
    L_over_D = CL_required/CD
    throttle_position = 0.6

    if engine['type'] == 0:
        thrust_force, fuel_flow , vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
    else:
        thrust_force = nn_unit_F.predict(scaler_F.transform([(altitude, mach, throttle_position)]))
        fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(altitude, mach, throttle_position)]))

    FnR = mass*GRAVITY/L_over_D

    step_throttle = 0.01
    throttle_position = 0.6
    total_thrust_force = 0

    while (total_thrust_force < FnR and throttle_position <= 1):

        if engine['type'] == 0:
            thrust_force, fuel_flow , vehicle = turbofan(
            altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
        else:
            thrust_force = nn_unit_F.predict(scaler_F.transform([(altitude, mach, throttle_position)]))
            fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(altitude, mach, throttle_position)]))

        TSFC = (fuel_flow*GRAVITY)/thrust_force
        total_thrust_force = aircraft['number_of_engines'] * thrust_force
        throttle_position = throttle_position+step_throttle

    L_over_D = CL_required/CD

    return TSFC, L_over_D, fuel_flow, throttle_position



import pickle

with open('Database/Family/40_to_100/all_dictionaries/'+str(58)+'.pkl', 'rb') as f:
    all_info_acft1 = pickle.load(f)

# with open('Database/Family/101_to_160/all_dictionaries/'+str(13)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family/161_to_220/all_dictionaries/'+str(0)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family_DD/40_to_100/all_dictionaries/'+str(28)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family_DD/101_to_160/all_dictionaries/'+str(28)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family_DD/161_to_220/all_dictionaries/'+str(0)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)
# vehicle = all_info_acft1['vehicle']

# aircraft = vehicle['aircraft']
# wing = vehicle['wing']
# engine = vehicle['engine']
# performance = vehicle['performance']
# operations = vehicle['operations']

# Payload_1 = aircraft['payload_weight']
# Payload_2 = aircraft['maximum_takeoff_weight'] - aircraft['operational_empty_weight'] - \
#     aircraft['crew_number']*100 - wing['fuel_capacity']
# Payload_3 = 0

# TOW_0 = aircraft['maximum_takeoff_weight'] - wing['fuel_capacity']
# TOW_1 = aircraft['maximum_takeoff_weight']
# TOW_2 = aircraft['maximum_takeoff_weight']
# TOW_3 = aircraft['operational_empty_weight'] + \
#     aircraft['crew_number']*100 + wing['fuel_capacity']

# Fuel_1 = aircraft['maximum_takeoff_weight'] - aircraft['operational_empty_weight'] - \
#     aircraft['crew_number']*100 - aircraft['payload_weight']
# Fuel_2 = wing['fuel_capacity']
# Fuel_3 = wing['fuel_capacity']

# # ---------------- Payload range  ----------------------


# # Point 1
# Range_1 = payload_range(TOW_1, Fuel_1, vehicle)

# # Point 2
# Range_2 = payload_range(TOW_2, Fuel_2, vehicle)

# # Point 3
# Range_3 = payload_range(TOW_3, Fuel_3, vehicle)

# Range_des = performance['range']

# ranges = [0, Range_1, Range_2, Range_3]
# payloads = [Payload_1, Payload_1, Payload_2, Payload_3]
# fuels = [0, Fuel_1, Fuel_2, Fuel_3]
# TOWs = [TOW_0, TOW_1, TOW_2, TOW_3]

# payloads_pct = [(x/aircraft['payload_weight'])*100 for x in payloads]
# fuels_pct = [(x/wing['fuel_capacity'])*100 for x in fuels]
# TOWs_pct = [(x/aircraft['maximum_takeoff_weight'])*100 for x in TOWs]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(ranges ,payloads,'k-')
# # ax.plot(ranges ,fuels_pct,'r--', label = 'Fuel load')
# # ax.plot(ranges ,TOWs_pct,'b--', label= 'TOW')
# ax.set_ylim(bottom=0)
# ax.set_xlim(left=0)
# ax.grid()
# ax.legend()
# ax.set_xlabel('Range [nm]')
# ax.set_ylabel('Payload [kg]')

# plt.show()
