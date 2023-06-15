import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv('plots/AirTNNvsGNN_snr_delta.csv')

model_type = 'airgnn'

df = df[df['model'] == model_type]

# Group by 'model/backbone/snr_db', 'model/backbone/delta' and calculate the mean of 'val/acc_best'
grouped_df = df.groupby(['model/backbone/snr_db', 'model/backbone/delta'])['val/acc_best'].mean().reset_index()

# Pivot the data to suit heatmap format
pivot_df = grouped_df.pivot('model/backbone/snr_db', 'model/backbone/delta', 'val/acc_best')

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, cmap='YlGnBu', annot=True)

plt.title(f'Heatmap of Mean val/acc_best - {model_type}')
plt.xlabel('model/backbone/delta')
plt.ylabel('model/backbone/snr_db')

# Invert Y-axis
plt.gca().invert_yaxis()

plt.savefig(f'plots/heat_map_{model_type}.png', dpi=300)

plt.show()
