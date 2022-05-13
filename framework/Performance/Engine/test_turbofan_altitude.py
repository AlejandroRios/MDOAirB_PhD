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
n = 11
M0 = np.linspace(0.1,0.85,n)
altitude = np.linspace(0,42000,n)
throttle_position = np.linspace(0.01,1,n)



FC_vec_ANN = []
fuel_flow_vec_ANN = []

FC_vec_model = []
F_vec_model = []
fuel_flow_vec_model = []

mach_vec = []
altitude_vec = []
engine = vehicle['engine']

fan_pressure_ratio = 1.5
compressor_pressure_ratio = 25
bypass_ratio = 4
fan_diameter = 1.5
turbine_inlet_temperature = 1450

for i in M0:
    for j in altitude:

        thrust_force, fuel_flow , vehicle = turbofan(
            j, i,1, vehicle)  # force [N], fuel flow [kg/hr]
        
        FC_vec_model.append(fuel_flow)


        
        # F_model, fuel_flow = PW120model(i,j,1)
        # FC_vec_model.append(float(fuel_flow))
        # F_vec_model.append(float(F_model))

        mach_vec.append(i)
    # altitude_vec.append(j)


FC_vec_model = np.reshape(FC_vec_model, (n,n))
mach_vec = np.reshape(mach_vec, (n,n))
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend',fontsize=12) # using a size in points
plt.rc('legend',fontsize='medium') # using a named size
plt.rc('axes',labelsize=12, titlesize=12) # using a size in points

fig1 = plt.figure(figsize=(10, 9))
ax1 = fig1.add_subplot(1, 1, 1)

# plt.plot(altitude,F_vec,'-')
# plt.plot(x[0:,0],y1,'o',alpha=0.5)
# plt.show()


ax1.plot(mach_vec,FC_vec_model,'-',alpha=0.5,linewidth=2)

# ax1.plot(x[0:,0],y1, 'kx',label='linear regression',linewidth=2)

ax1.set_xlabel('Mach number')
ax1.set_ylabel('TSFC [Kg/hr/N]')
# ax1.set_title('Activation function: ReLU')

ax1.set_xlim([None,None])
ax1.set_ylim([None,None])


# first_leg = mpatches.Patch(label=altitude_vec)

# plt.legend(handles=[first_leg])

plt.grid(True)
plt.show()