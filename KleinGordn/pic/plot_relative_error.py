import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV files
df = pd.read_csv('relative_errors.csv')

# Convert 'network_structure' to a classification type with a specific order
network_order = ["3x30", "3x50", "3x70", "3x90", "5x30", "5x50", "5x70", "5x90", "7x30", "7x50", "7x70", "7x90"]
df['network_structure'] = pd.Categorical(df['network_structure'], categories=network_order, ordered=True)

#Define tags and line types for each model type
markers = {"PINN": "o", "IFNN-PINN": "s", "A-PINN": "D", "H-PINN": "^"}
linestyles = {"PINN": ":", "IFNN-PINN": "--", "A-PINN": "-.", "H-PINN": "-"}

# Setting the drawing style
sns.set(style="whitegrid")

# Create a line chart
plt.figure(figsize=(12, 8), dpi=300)

for model_type, marker in markers.items():
    linestyle = linestyles[model_type]
    subset = df[df['model_type'] == model_type]
    sns.lineplot(
        data=subset,
        x="network_structure",
        y="relative_error",
        marker=marker,
        linestyle=linestyle,
        label=model_type,
        linewidth=2.5
    )

# Set fonts and labels
plt.xlabel("Network Structure", fontsize=14, fontname='Arial')
plt.ylabel("Relative Error", fontsize=14, fontname='Arial')
plt.yscale('log')
plt.xticks(rotation=45, fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')
plt.legend(title='Model Type', fontsize=12, title_fontsize=14, loc='upper right', frameon=False)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.legend(loc='upper right')

plt.tight_layout()

#Saving the fig
plt.savefig('Relative_Error_Comparison_for_Different_Network_Structures_and_Models.pdf')
plt.show()