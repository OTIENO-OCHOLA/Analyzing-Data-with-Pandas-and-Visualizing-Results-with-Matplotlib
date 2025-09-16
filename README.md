# Analyzing-Data-with-Pandas-and-Visualizing-Results-with-Matplotlib

Objective For this Assignment:  To load and analyze a dataset using the pandas library in Python. To create simple plots and charts with the matplotlib library for visualizing the data.

# Import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better looking plots

plt.style.use('seaborn-v0_8')

# Load the Iris dataset

# Note: You can also load from a local file using pd.read_csv('path/to/file.csv')

url = "<https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data>"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, names=column_names)

# Display first few rows

print("First 5 rows of the dataset:")
print(iris_df.head())
print("\n" + "="*50 + "\n")

# Explore dataset structure

print("Dataset information:")
print(iris_df.info())
print("\n" + "="*50 + "\n")

print("Dataset shape:", iris_df.shape)
print("\nColumn names:", iris_df.columns.tolist())
print("\n" + "="*50 + "\n")

# Check for missing values

print("Missing values in each column:")
print(iris_df.isnull().sum())
print("\n" + "="*50 + "\n")

# Since there are no missing values in this dataset, no cleaning is needed

# But for demonstration, here's how you would handle missing values

if iris_df.isnull().sum().sum() > 0:
    print("Cleaning dataset...")
    # Option 1: Drop rows with missing values
    iris_df = iris_df.dropna()

    # Option 2: Fill missing values (example for numerical columns)
    # iris_df = iris_df.fillna(iris_df.mean())
else:
    print("No missing values found. Dataset is clean!")
    # Basic statistics for numerical columns
print("Basic statistics for numerical columns:")
print(iris_df.describe())
print("\n" + "="*50 + "\n")

# Statistics by species

print("Statistics grouped by species:")
grouped_stats = iris_df.groupby('species').describe()
print(grouped_stats)
print("\n" + "="*50 + "\n")

# Mean of numerical columns by species

print("Mean values by species:")
species_means = iris_df.groupby('species').mean()
print(species_means)
print("\n" + "="*50 + "\n")

# Interesting findings

print("Interesting Findings:")
print("1. Setosa species has the smallest petals but largest sepals")
print("2. Virginica has the largest petals on average")
print("3. Versicolor is intermediate in most measurements")
print("4. Petal measurements show more variation than sepal measurements")

# Create a 2x2 grid of subplots

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Multiple Visualizations', fontsize=16, fontweight='bold')

# 1. Line Chart - Trends across samples (simulating time series)

# Since we don't have time data, we'll use index as x-axis

axes[0, 0].plot(iris_df.index, iris_df['sepal_length'], label='Sepal Length', alpha=0.7)
axes[0, 0].plot(iris_df.index, iris_df['petal_length'], label='Petal Length', alpha=0.7)
axes[0, 0].set_title('Line Chart: Sepal and Petal Length Trends Across Samples')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar Chart - Comparison of average measurements by species

species_colors = {'Iris-setosa': 'skyblue', 'Iris-versicolor': 'lightgreen', 'Iris-virginica': 'salmon'}
measurements = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
x_pos = np.arange(len(measurements))
width = 0.25

for i, species in enumerate(iris_df['species'].unique()):
    species_data = iris_df[iris_df['species'] == species][measurements].mean()
    axes[0, 1].bar(x_pos + i*width, species_data, width,
                  label=species, color=species_colors[species], alpha=0.8)

axes[0, 1].set_title('Bar Chart: Average Measurements by Species')
axes[0, 1].set_xlabel('Measurement Type')
axes[0, 1].set_ylabel('Average Measurement (cm)')
axes[0, 1].set_xticks(x_pos + width)
axes[0, 1].set_xticklabels(['Sepal\nLength', 'Sepal\nWidth', 'Petal\nLength', 'Petal\nWidth'])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Histogram - Distribution of sepal length

axes[1, 0].hist(iris_df['sepal_length'], bins=15, alpha=0.7, color='lightblue', edgecolor='black')
axes[1, 0].axvline(iris_df['sepal_length'].mean(), color='red', linestyle='--',
                  label=f'Mean: {iris_df["sepal_length"].mean():.2f}cm')
axes[1, 0].set_title('Histogram: Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter Plot - Relationship between sepal length and petal length

colors = iris_df['species'].map(species_colors)
scatter = axes[1, 1].scatter(iris_df['sepal_length'], iris_df['petal_length'],
                            c=colors, alpha=0.7, s=50)
axes[1, 1].set_title('Scatter Plot: Sepal Length vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')

# Create legend for scatter plot

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[species],
                         markersize=10, label=species) for species in species_colors.keys()]
axes[1, 1].legend(handles=legend_elements)
axes[1, 1].grid(True, alpha=0.3)

# Add correlation coefficient

correlation = iris_df['sepal_length'].corr(iris_df['petal_length'])
axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
               transform=axes[1, 1].transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Additional visualization: Box plot by species

plt.figure(figsize=(12, 6))
iris_df.boxplot(by='species', column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
               grid=False)
plt.suptitle('')  # Remove automatic title
plt.title('Box Plot: Measurement Distributions by Species', fontsize=14, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Measurement (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("SUMMARY OF FINDINGS:")
print("=" * 50)
print("1. DATA QUALITY: The Iris dataset is clean with no missing values")
print("2. SPECIES DIFFERENCES: Clear morphological differences exist between the three iris species")
print("3. CORRELATIONS: Strong positive correlation (0.87) between sepal length and petal length")
print("4. DISTRIBUTIONS: Sepal length follows a roughly normal distribution")
print("5. PATTERNS: Setosa is most distinct, while versicolor and virginica show some overlap")
print("\nThe visualizations effectively demonstrate:")
print("   - Trends across samples (Line Chart)")
print("   - Species comparisons (Bar Chart)")
print("   - Data distribution (Histogram)")
print("   - Variable relationships (Scatter Plot)")
