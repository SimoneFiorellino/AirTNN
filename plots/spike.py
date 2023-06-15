import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df1 = pd.read_csv('plots/spike.csv')

# Group by 'spike' and calculate mean and standard deviation
grouped1 = df1.groupby('dataset.spike')['val/acc_best'].agg(['mean', 'std'])

# Plot mean with standard deviation bars for both datasets
plt.errorbar(grouped1.index, grouped1['mean'], yerr=grouped1['std'], 
             fmt='o', label='AirTNN Mean with STD bars', color='blue')

# Plot line connecting the means for both datasets
plt.plot(grouped1.index, grouped1['mean'], label='AirTNN', color='orange')

# Add labels
plt.xlabel('spike')
plt.ylabel('accuracy')
plt.title('Mean and STD of Accuracy Grouped by spike')

# Add legend
plt.legend()

plt.savefig('plots/spike.png', dpi=300)

# Show the plot
plt.show()