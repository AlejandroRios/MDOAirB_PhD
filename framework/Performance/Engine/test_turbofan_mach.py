# from joblib import dump, load
import numpy as np
from framework.Performance.Engine.Turboprop.PW120model import PW120model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from framework.Performance.Engine.engine_performance import turbofan
import pickle

with open('Database/Family/161_to_220/all_dictionaries/'+str(1)+'.pkl', 'rb') as f:
    all_info_acft3 = pickle.load(f)


vehicle = all_info_acft3['vehicle']
n = 21
M0 = np.linspace(0.1,0.85,n)
altitude = np.linspace(0,42000,n)
throttle_position = np.linspace(0.01,1,n)



FC_vec_ANN = []
fuel_flow_vec_ANN = []

FC_vec_model = []
F_vec_model = []
fuel_flow_vec_model = []

altitude_vec = []
engine = vehicle['engine']

fan_pressure_ratio = 2
compressor_pressure_ratio = 30
bypass_ratio = 6
fan_diameter = 2
turbine_inlet_temperature = 1500

for i in altitude:
    for j in M0:

        thrust_force, fuel_flow , vehicle = turbofan(
            i, j,1, vehicle)  # force [N], fuel flow [kg/hr]
        
        FC_vec_model.append(thrust_force)


        
        # F_model, fuel_flow = PW120model(i,j,1)
        # FC_vec_model.append(float(fuel_flow))
        # F_vec_model.append(float(F_model))

        altitude_vec.append(i)


FC_vec_model = np.reshape(FC_vec_model, (n,n))
altitude_vec = np.reshape(altitude_vec, (n,n))
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig1 = plt.figure(figsize=(10, 9))
ax1 = fig1.add_subplot(1, 1, 1)

# plt.plot(altitude,F_vec,'-')
# plt.plot(x[0:,0],y1,'o',alpha=0.5)
# plt.show()


ax1.plot(altitude_vec,FC_vec_model,'-',alpha=0.5,label='Turbofan model')

# ax1.plot(x[0:,0],y1, 'kx',label='linear regression',linewidth=2)

ax1.set_xlabel('Altitude [ft]')
ax1.set_ylabel('FC [Kg/hr]')
# ax1.set_title('Activation function: ReLU')

ax1.set_xlim([None,None])
ax1.set_ylim([None,None])


first_leg = mpatches.Patch(label='Turbofan model')

plt.legend(handles=[first_leg])

plt.grid(True)
plt.show()