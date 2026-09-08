from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Create boxplot and save it as a PNG file
import matplotlib.pyplot as plt

# Create a boxplot for 'MedHouseVal'
plt.figure(figsize=(8, 6))
plt.boxplot(df['MedHouseVal'])
plt.title('Boxplot of Median House Value')
plt.ylabel('Median House Value')
plt.xticks([]) # Hide x-axis ticks for a single boxplot
# plt.ylim(4.5, 5.1) # Zooming to the 

# Save the boxplot
plt.savefig('figs/boxplot.png', bbox_inches='tight', dpi=300)
print("Boxplot saved as 'figs/boxplot.png'")