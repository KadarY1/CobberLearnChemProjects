"""
Iris Clustering with KMeans
--------------------------------
This script:
1. Loads the Iris dataset from sklearn
2. Creates a DataFrame with only feature columns
3. Applies KMeans clustering (k=3)
4. Visualizes results in 2D and 3D plots
"""

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# -------------------------------
# Step 1: Load the Iris dataset
# -------------------------------
iris = load_iris()

# Create a DataFrame using ONLY the feature data (exclude target)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# -------------------------------
# Step 2: Apply KMeans Clustering
# -------------------------------
# Initialize model with 3 clusters (since we know there are 3 species)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit model and assign cluster labels
df['cluster'] = kmeans.fit_predict(df)

# -------------------------------
# Step 3: 2D Scatter Plot
# Petal Length vs Petal Width
# -------------------------------
plt.figure(figsize=(8, 6))

# Scatter plot colored by cluster
scatter = plt.scatter(
    df['petal length (cm)'],
    df['petal width (cm)'],
    c=df['cluster']
)

# Add labels and title
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('KMeans Clustering of Iris Dataset (2D)')

# Create legend from scatter colors
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()

# -------------------------------
# Step 4: 3D Scatter Plot
# Using three features
# -------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot using three features
ax.scatter(
    df['sepal length (cm)'],
    df['petal length (cm)'],
    df['petal width (cm)'],
    c=df['cluster']
)

# Add labels and title
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_zlabel('Petal Width (cm)')
ax.set_title('KMeans Clustering of Iris Dataset (3D)')

plt.show()

# -------------------------------
# Notes on Organization & Clarity
# -------------------------------
"""
- Code is divided into clear sections (loading, modeling, visualization)
- Comments explain each step
- Variable names are descriptive
- Plots are labeled for easy interpretation
"""