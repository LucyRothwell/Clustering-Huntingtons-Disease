import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure() # Create blank figure
fig.subplots_adjust(top=0.8) # ?
ax1 = fig.add_subplot(211) # Adds X and Y axis values 0-1
ax1.set_ylabel('volts') # Adds Y axis label
ax1.set_title('a sine wave') # Adds figure title

t = np.arange(0.0, 1.0, 0.01) # Creating X data
s = np.sin(2 * np.pi * t) # Creating y data
line, = ax1.plot(t, s, lw=2) # Plotting data on figure

# Fixing random state for reproducibility
np.random.seed(19680801)

ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
n, bins, patches = ax2.hist(np.random.randn(1000), 50)
ax2.set_xlabel('time (s)')

plt.show()