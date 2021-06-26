# Plotting practice

from matplotlib import pyplot as plt

# DEFINE DATA
x = [2,3, 4,5]
y = [102,105,107,109]

# PLOT THE DATA
plt.plot(x,y)

# LABEL THE CHART
plt.title("Chart of joy") # TITLE
plt.ylabel("Y axis") # LABEL Y AXIS
plt.xlabel("X axis") # LABEL X AXIS

# STYLE THE CHART
from matplotlib import style
# Style is a folder we created which holds style templates we (or someone else) created.
# By importing them we can use the same styles for all of our plots and histograms etc. GUIDE ON HOW TO CREEATE STYLE FOLDER: https://youtu.be/sCaGYsEYy-k?list=PLQVvvaa0QuDe8XSftW-RAxdo6OmaeL85M

plt.plot(x,y, "g", linewidth=5)
# "g" = green
# linewidth - literally thickness of line on plot


