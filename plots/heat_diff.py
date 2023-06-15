import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv('plots/AirTNNvsGNN_snr_delta.csv')

# wrt
score = 'test/acc'

# Filter rows for 'airtnn' and 'airgnn' models
df_airtnn = df[df['model'] == 'airtnn']
df_airgnn = df[df['model'] == 'airgnn']

# Group by 'model/backbone/snr_db', 'model/backbone/delta' and calculate the mean of score for each model
grouped_df_airtnn = df_airtnn.groupby(['model/backbone/snr_db', 'model/backbone/delta'])[score].mean().reset_index()
grouped_df_airgnn = df_airgnn.groupby(['model/backbone/snr_db', 'model/backbone/delta'])[score].mean().reset_index()

# Pivot the data to suit heatmap format for each model
pivot_df_airtnn = grouped_df_airtnn.pivot('model/backbone/snr_db', 'model/backbone/delta', score)
pivot_df_airgnn = grouped_df_airgnn.pivot('model/backbone/snr_db', 'model/backbone/delta', score)

# Calculate the difference between the two models
diff_df = pivot_df_airtnn - pivot_df_airgnn

# Create a heatmap for the difference
plt.figure(figsize=(10, 8))
sns.heatmap(diff_df, cmap='coolwarm', center=0, annot=True)

plt.title('Heatmap of Differences in Mean val/acc_best')
plt.xlabel('model/backbone/delta')
plt.ylabel('model/backbone/snr_db')

# Invert Y-axis
plt.gca().invert_yaxis()

plt.savefig(f'plots/heat_map_difference_test.png', dpi=300)

plt.show()
