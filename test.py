import numpy as np
import matplotlib.pyplot as plt
import afm_functions as afm

data = afm.import_data(r"C:\Users\poeda\OneDrive\Documents\Luo Group\AFM Images\MAXYMUSJuly2025\Sample 16\Main 0°\MAXYMUS #16 0deg profiles.csv")

# afm.plotly_graph(data)
# plt.show()

avg = afm.threshold_averaging(data, 0.1e-8)
afm.subplot_heights(avg, True, [(2,3),(3,2)])