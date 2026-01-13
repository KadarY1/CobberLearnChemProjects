import matplotlib.pyplot as plt

# Data
num_carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
boiling_points = [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.6, 150.8, 174.1]

# Create figure with better aspect ratio
plt.figure(figsize=(9, 6))

# Scatter plot with custom marker and color styling
plt.scatter(
    num_carbons,
    boiling_points,
    s=140,                 # marker sizes
    marker='h',            # hexagon marker (feels more "chemical")
    c=boiling_points,      # color mapped to boiling point
    cmap='plasma',         # presentation-friendly colormap
    edgecolors='black',
    linewidth=0.8,
    alpha=0.9
)

# Add a smooth connecting line for trend emphasis
plt.plot(
    num_carbons,
    boiling_points,
    linestyle='--',
    linewidth=1.5,
    alpha=0.6
)

# Titles and labels
plt.title(
    "Boiling Point vs Number of Carbons\n(Linear Alkanes)",
    fontsize=16,
    fontweight='bold'
)
plt.xlabel("Number of Carbons", fontsize=13)
plt.ylabel("Boiling Point (°C)", fontsize=13)

# Improve grid appearance
plt.grid(True, linestyle='--', alpha=0.4)

# Add colorbar for clarity
cbar = plt.colorbar()
cbar.set_label("Boiling Point (°C)", fontsize=11)

# Clean up spines for a modern look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
