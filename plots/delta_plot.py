import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df1 = pd.read_csv('plots/airtnn_delta_sweep_airgnn_k1.csv')
df2 = pd.read_csv('plots/airtnn_delta_sweep_airgnn_k5.csv')  # replace with your second CSV file

# Group by 'snr_db' and calculate mean and standard deviation
grouped1 = df1.groupby('model/backbone/delta')['val/acc_best'].agg(['mean', 'std'])
grouped2 = df2.groupby('model/backbone/delta')['val/acc_best'].agg(['mean', 'std'])  # do the same for second dataset

# Plot mean with standard deviation bars for both datasets
plt.errorbar(grouped1.index, grouped1['mean'], yerr=grouped1['std'], 
             fmt='o', label='AirGNN_k1 Mean with STD bars', color='blue')
plt.errorbar(grouped2.index, grouped2['mean'], yerr=grouped2['std'], 
             fmt='o', label='AirGNN_k5 Mean with STD bars', color='green')  # adjust color as needed

# Plot line connecting the means for both datasets
plt.plot(grouped1.index, grouped1['mean'], label='AirGNN_k1', color='red')
plt.plot(grouped2.index, grouped2['mean'], label='AirGNN_k5', color='orange')  # adjust color as needed

# Add labels
plt.xlabel('delta')
plt.ylabel('accuracy')
plt.title('Mean and STD of Accuracy Grouped by delta')

# Add legend
plt.legend()

plt.savefig('plots/delta_comparison_plot_k1vsk5.png', dpi=300)

# Show the plot
plt.show()