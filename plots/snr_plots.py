import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('plots/snrdb_acc.csv')

# Get the unique models and number of parameters in the DataFrame
unique_models = df['model'].unique()
unique_num_params = df['num_params'].unique()

# Create a list of colors for the models. Adjust as needed.
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

# Create a plot for each model
index = 0
for model in unique_models:
    for num_param in unique_num_params:
        sub_df = df[(df['model'] == model) & (df['num_params'] == num_param)]
        if not sub_df.empty:
            grouped = sub_df.groupby('model/backbone/snr_db')['test/acc'].agg(['mean', 'std'])
            model_with_params = model + " " + str(num_param // 10**3) + 'K' + " params"
            plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                         fmt='o', color=colors[index % len(colors)])
            plt.plot(grouped.index, grouped['mean'], label=model_with_params, color=colors[index % len(colors)])
            index += 1

plt.axhline(y=0.97, color='r', linestyle='--', label='Models without fading and noise')


# Add labels
plt.xlabel('snr_db')
plt.ylabel('accuracy')
plt.title('Mean and STD of Accuracy Grouped by snr_db')

# Add legend
plt.legend()

# Add grid
plt.grid(True)

plt.savefig('plots/snr_db_comparison_plot.png', dpi=300)

# Show the plot
plt.show()
