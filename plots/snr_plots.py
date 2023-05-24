import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df1 = pd.read_csv('plots/snrdb_acc.csv')
df2 = pd.read_csv('plots/snrdb_acc_airgnn.csv')  # replace with your second CSV file

# Group by 'snr_db' and calculate mean and standard deviation
grouped1 = df1.groupby('model.backbone.snr_db')['val/acc_best'].agg(['mean', 'std'])
grouped2 = df2.groupby('model.backbone.snr_db')['val/acc_best'].agg(['mean', 'std'])  # do the same for second dataset

# Plot mean with standard deviation bars for both datasets
plt.errorbar(grouped1.index, grouped1['mean'], yerr=grouped1['std'], 
             fmt='o', label='AirTNN Mean with STD bars', color='blue')
plt.errorbar(grouped2.index, grouped2['mean'], yerr=grouped2['std'], 
             fmt='o', label='AirGNN Mean with STD bars', color='green')  # adjust color as needed

# Plot line connecting the means for both datasets
plt.plot(grouped1.index, grouped1['mean'], label='AirTNN', color='red')
plt.plot(grouped2.index, grouped2['mean'], label='AirGNN', color='orange')  # adjust color as needed

# Add labels
plt.xlabel('snr_db')
plt.ylabel('accuracy')
plt.title('Mean and STD of Accuracy Grouped by snr_db')

# Add legend
plt.legend()

plt.savefig('plots/snr_db_comparison_plot.png', dpi=300)

# Show the plot
plt.show()