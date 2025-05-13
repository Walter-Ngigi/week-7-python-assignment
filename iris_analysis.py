
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn style
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load the dataset (using Iris from sklearn for simplicity)
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing values per column:")
    print(df.isnull().sum())

except FileNotFoundError:
    print("Dataset not found.")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

print("\nMean values per species:")
print(df.groupby("species").mean())

# Task 3: Data Visualization

# Line Plot: Simulated time-series by treating row index as time
plt.figure(figsize=(10, 5))
for species in df['species'].unique():
    sns.lineplot(data=df[df['species'] == species].reset_index(), x='index', y='sepal length (cm)', label=species)
plt.title("Line Chart: Sepal Length over Index (Simulated Time)")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("line_plot.png")
plt.close()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator='mean')
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.close()

# Histogram: Sepal Width Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], kde=True, bins=15)
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()

# Findings and Observations
print("\nObservations:")
print("- Setosa tends to have smaller petal length and sepal length.")
print("- Versicolor and Virginica show overlapping values, but Virginica tends to be larger.")
print("- Sepal width distribution is slightly right-skewed.")
print("- Scatter plot shows clear clustering by species.")

print("\nPlots saved: line_plot.png, bar_chart.png, histogram.png, scatter_plot.png")
