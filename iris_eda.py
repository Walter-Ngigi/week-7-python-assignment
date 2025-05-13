
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = iris.target
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
except Exception as e:
    print("Error loading dataset:", e)

# Task 2: Basic Data Analysis
print("\nSummary Statistics:")
print(df.describe())

grouped = df.groupby('target').mean()
print("\nMean values by target class:")
print(grouped)

# Task 3: Data Visualization
sns.set(style="whitegrid")

# Line chart
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.savefig("line_chart.png")
plt.close()

# Bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette="muted")
plt.title('Average Petal Length by Class')
plt.xlabel('Class')
plt.ylabel('Petal Length (cm)')
plt.savefig("bar_chart.png")
plt.close()

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.savefig("histogram.png")
plt.close()

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette="deep")
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.savefig("scatter_plot.png")
plt.close()
