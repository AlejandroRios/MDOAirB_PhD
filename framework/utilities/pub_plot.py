import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Activation function: ReLU')

ax.set_xlim([0,None])
ax.set_ylim([0,None])

plt.grid(True)
plt.show()