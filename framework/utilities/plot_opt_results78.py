import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
with open(r"Database/Results_Multi_Optim/functions/functions_multi_obj_DD_profit_cost.pkl", "rb") as input_file:
    df_DD = pickle.load(input_file)

with open(r"Database/Results_Multi_Optim/functions/functions_multi_obj_GCD_profit_cost.pkl", "rb") as input_file:
    df_GCD = pickle.load(input_file)

with open(r"Database/Results_Multi_Optim/functions/case7_profit_pareto.pkl", "rb") as input_file:
    F1_GCD = pickle.load(input_file)
with open(r"Database/Results_Multi_Optim/functions/case7_cost_pareto.pkl", "rb") as input_file:
    F2_GCD = pickle.load(input_file)

# with open(r"Database/Results_Multi_Optim/functions/case6_profit_pareto.pkl", "rb") as input_file:
#     F1_DD = pickle.load(input_file)
# with open(r"Database/Results_Multi_Optim/functions/case6_C02_pareto.pkl", "rb") as input_file:
#     F2_DD = pickle.load(input_file)



print('min CO2',np.min(F2_GCD))
print('max prof',np.min(F1_GCD))

# F1_DD = pd.DataFrame(F1_DD, columns=['F1'])
# F2_DD = pd.DataFrame(F2_DD, columns=['F2'])

F1_GCD = pd.DataFrame(F1_GCD, columns=['F1'])
F2_GCD = pd.DataFrame(F2_GCD, columns=['F2'])
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend',fontsize=12) # using a size in points
plt.rc('legend',fontsize='medium') # using a named size
plt.rc('axes',labelsize=12, titlesize=12) # using a size in points

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('Profit [US$]')
ax.set_ylabel('CO2 efficiecy [kg/nm]')


ax.scatter(df_GCD['X1'], df_GCD['X2'],s=30, facecolors='none', edgecolors='grey',alpha = 0.5,label='GCD solutions')
ax.scatter(df_DD['X1'], df_DD['X2'],s=30, facecolors='none', edgecolors='skyblue',alpha = 0.5,label='DD solutions')
ax.scatter(F1_GCD, F2_GCD, s=50, facecolors='none',marker= '^',edgecolors='black')
ax.scatter(F1_DD, F2_DD, s=50, facecolors='none',marker= 's',edgecolors='black')
ax.set_title("Objective Space")

plt.legend(loc='upper left')

plt.show()