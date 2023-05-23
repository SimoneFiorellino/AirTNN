import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('snrdb_acc.csv')

# Group by 'snr_db' and calculate mean and standard deviation
grouped = df.groupby('model.backbone.snr_db')['val/acc_best'].agg(['mean', 'std'])

# Plot mean with standard deviation bars
plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
             fmt='o', label='Mean with STD bars', color='blue')

# Plot line connecting the means
plt.plot(grouped.index, grouped['mean'], label='Mean trend line', color='red')

# Add labels
plt.xlabel('snr_db')
plt.ylabel('accuracy')
plt.title('Mean and STD of Accuracy Grouped by snr_db')

# Add legend
plt.legend()

# Show the plot
plt.show()