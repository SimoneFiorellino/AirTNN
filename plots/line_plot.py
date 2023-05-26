import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv('plots/heat_map.csv')

# Calculate the mean and standard deviation of 'val/acc_best'
df_stats = df.groupby(['model/backbone/snr_db', 'model/backbone/delta'])['val/acc_best'].agg(['mean', 'std']).reset_index()

# Create a figure
plt.figure(figsize=(10, 8))

# List of unique deltas
deltas = df_stats['model/backbone/delta'].unique()

# Create error bar for each delta
for delta in deltas:
    subset = df_stats[df_stats['model/backbone/delta'] == delta]
    plt.errorbar(x=subset['model/backbone/snr_db'], y=subset['mean'], yerr=subset['std'], label=f'delta={delta}')

plt.title('Mean and Standard Deviation of val/acc_best')
plt.xlabel('model/backbone/snr_db')
plt.ylabel('val/acc_best')

plt.legend()

plt.savefig('plots/error_bar_plot.png', dpi=300)

plt.show()

