# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# df = pd.read_csv('plots/delta_acc.csv')

# # Filter DataFrame based on 'model' and group by 'delta'
# grouped1 = df[df['model'] == 'airtnn'].groupby('model.backbone.delta')['test/acc'].agg(['mean', 'std'])
# grouped2 = df[df['model'] == 'airgnn'].groupby('model.backbone.delta')['test/acc'].agg(['mean', 'std'])

# # Plot mean with standard deviation bars for both datasets
# plt.errorbar(grouped1.index, grouped1['mean'], yerr=grouped1['std'], 
#              fmt='o', label='AirTNN Mean with STD bars', color='blue')
# plt.errorbar(grouped2.index, grouped2['mean'], yerr=grouped2['std'], 
#              fmt='o', label='AirGNN Mean with STD bars', color='green')

# # Plot line connecting the means for both datasets
# plt.plot(grouped1.index, grouped1['mean'], label='AirTNN', color='red')
# plt.plot(grouped2.index, grouped2['mean'], label='AirGNN', color='orange')

# # Add labels
# plt.xlabel('delta')
# plt.ylabel('accuracy')
# plt.title('Mean and STD of Accuracy Grouped by delta')

# # Add legend
# plt.legend()

# # Add grid
# plt.grid(True)

# plt.savefig('plots/delta_comparison_plot.png', dpi=300)

# # Show the plot
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Load the CSV file
df = pd.read_csv('plots/delta_acc.csv')

# Get unique models
models = df['model'].unique()

# Create a list of colors
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

# Placeholder for legend elements
legend_elements = []

for model in models:
    # Filter by model type
    df_model = df[df['model'] == model]
    
    # Get unique combinations of hyperparameters for each model
    hyperparameters = df_model[['model/backbone/k', 'model/backbone/hidden_dim']].drop_duplicates()
    
    for idx, row in hyperparameters.iterrows():
        # Get next color
        color = next(colors)
        
        # Filter DataFrame based on hyperparameters
        df_hyper = df_model[(df_model['model/backbone/k'] == row['model/backbone/k']) & 
                            (df_model['model/backbone/hidden_dim'] == row['model/backbone/hidden_dim'])]
        
        # Group by 'delta' and compute mean and std
        grouped = df_hyper.groupby('model.backbone.delta')['test/acc'].agg(['mean', 'std'])
        
        # Plot mean with standard deviation bars
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                     fmt='o', label=f'{model} {row["model/backbone/k"]}/{row["model/backbone/hidden_dim"]} Mean with STD bars', 
                     color=color)
        
        # Plot line connecting the means
        plt.plot(grouped.index, grouped['mean'], label=f'{model} {row["model/backbone/k"]}/{row["model/backbone/hidden_dim"]}', 
                 color=color)
        
        legend_elements.append(f'{model} {row["model/backbone/k"]}/{row["model/backbone/hidden_dim"]}')

# Add labels
plt.xlabel('delta')
plt.ylabel('accuracy')
plt.title('Mean and STD of Accuracy Grouped by delta')

# Add legend
plt.legend(legend_elements, loc='lower right')

# Add grid
plt.grid(True)

plt.savefig('plots/delta_comparison_plot.png', dpi=300)

# Show the plot
plt.show()