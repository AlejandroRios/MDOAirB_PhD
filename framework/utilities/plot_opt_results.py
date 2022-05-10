import pickle
import matplotlib.pyplot as plt
import pandas as pd
with open(r"Database/Results_Multi_Optim/functions_multi_obj_DD_profit_CO2.pkl", "rb") as input_file:
    df_DD = pickle.load(input_file)

with open(r"Database/Results_Multi_Optim/functions_multi_obj_GCD_profit_CO2.pkl", "rb") as input_file:
    df_GCD = pickle.load(input_file)



F1_DD = [-2.14200200e+06, 
-8.11608000e+05, 
-1.44075600e+06, 
-9.50757000e+05, 
-2.06891000e+06, 
-1.74387900e+06, 
-1.70501400e+06, 
-1.69872100e+06, 
-1.64215400e+06, 
-2.13352800e+06, 
-1.53416200e+06, 
-8.53075000e+05, 
-8.90107000e+05, 
-2.07936200e+06, 
-2.13599800e+06, 
-1.67990000e+06, 
-2.11310000e+06, 
-1.63941800e+06, 
-2.12184900e+06, 
-1.61633000e+06, 
-1.55569400e+06, 
-1.57824300e+06, 
-1.55184700e+06, 
-9.50396000e+05, 
-8.29710000e+05, 
-1.69638300e+06, 
-1.62642400e+06, 
-9.32582000e+05, 
-8.43948000e+05, 
-1.57482400e+06, 
-8.99844000e+05, 
-1.55705700e+06, 
-1.69125700e+06, 
-1.73268300e+06, 
-9.29047000e+05, 
-1.56940800e+06, 
-9.07407000e+05, 
-1.63496100e+06, 
-9.18166000e+05, 
-1.74213300e+06, 
-1.72100200e+06, 
-9.06471000e+05]
F2_DD = [4.52629816e-04,
1.61616943e-04,
2.21832980e-04,
2.20229335e-04,
3.98665377e-04,
3.65555214e-04,
3.49446909e-04,
3.18541103e-04,
2.93195857e-04,
4.27557745e-04,
2.24715340e-04,
1.71169529e-04,
1.86345487e-04,
4.05741378e-04,
4.49474060e-04,
3.01915120e-04,
4.12216428e-04,
2.75185328e-04,
4.19310232e-04,
2.63448391e-04,
2.36760703e-04,
2.59626009e-04,
2.25248459e-04,
2.11923395e-04,
1.62444198e-04,
3.08916513e-04,
2.68721631e-04,
2.08403795e-04,
1.69480943e-04,
2.55337116e-04,
1.86868155e-04,
2.43359219e-04,
3.05494782e-04,
3.58911037e-04,
2.02634709e-04,
2.49317267e-04,
1.98094640e-04,
2.73811153e-04,
2.01726760e-04,
3.62630290e-04,
3.56125976e-04,
1.89213058e-04]

F1_GCD = [-1.17526200e+06,
-2.31318300e+06,
-1.51347700e+06,
-1.32941600e+06,
-2.30102000e+06,
-1.74181600e+06,
-2.09662300e+06,
-2.09080600e+06,
-2.07362600e+06,
-2.09123400e+06,
-2.08862400e+06,
-1.21673100e+06,
-1.88846600e+06,
-1.18747200e+06,
-1.97117600e+06,
-1.84069400e+06,
-2.06597700e+06,
-1.79545900e+06,
-2.00038400e+06,
-2.30119300e+06,
-1.76235700e+06,
-1.99349600e+06,
-1.85858500e+06,
-2.03345500e+06,
-2.02131900e+06,
-1.75891800e+06,
-1.98123800e+06,
-1.19631400e+06,
-2.30722500e+06,
-1.85745700e+06,
-1.85649700e+06,
-1.78685000e+06,
-1.78641700e+06,
-1.78779100e+06,
-1.78042000e+06,
-2.01974500e+06,
-2.05480300e+06,
-1.77318300e+06,
-2.04791000e+06,
-1.78788300e+06,
-1.77570000e+06,
-1.75077800e+06]

F2_GCD = [1.17605071e-04,
3.76695989e-04,
1.59289928e-04,
1.50205565e-04,
3.58315155e-04,
1.64479169e-04,
3.55917027e-04,
3.17992494e-04,
2.85074876e-04,
3.54301111e-04,
3.07464067e-04,
1.46372534e-04,
2.25162098e-04,
1.41345727e-04,
2.29102853e-04,
2.10271524e-04,
2.68368924e-04,
2.03328696e-04,
2.46979120e-04,
3.69436374e-04,
1.77405053e-04,
2.37640697e-04,
2.21724816e-04,
2.61728091e-04,
2.50859381e-04,
1.67006823e-04,
2.35193915e-04,
1.45333651e-04,
3.73913460e-04,
2.16100984e-04,
2.12439325e-04,
1.92756630e-04,
1.88697132e-04,
1.97460922e-04,
1.85166559e-04,
2.49761772e-04,
2.66562442e-04,
1.79951114e-04,
2.64036357e-04,
2.01454167e-04,
1.82438958e-04,
1.64968605e-04]

F1_DD = pd.DataFrame(F1_DD, columns=['F1'])
F2_DD = pd.DataFrame(F2_DD, columns=['F2'])

F1_GCD = pd.DataFrame(F1_GCD, columns=['F1'])
F2_GCD = pd.DataFrame(F2_GCD, columns=['F2'])
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('Profit [US$]')
ax.set_ylabel('CO2 efficiecy [kg/nm]')


ax.scatter(df_GCD['X1'], df_GCD['X2'],s=30, facecolors='none', edgecolors='grey',alpha = 0.5)
ax.scatter(df_DD['X1'], df_DD['X2'],s=30, facecolors='none', edgecolors='skyblue',alpha = 0.5)
ax.scatter(F1_GCD, F2_GCD, s=50, facecolors='none',marker= '^',edgecolors='black')
ax.scatter(F1_DD, F2_DD, s=50, facecolors='none',marker= 's',edgecolors='black')
ax.set_title("Objective Space")

plt.show()