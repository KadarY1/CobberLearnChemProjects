import matplotlib.pyplot as plt
import numpy as np

# Data
num_carbons = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
boiling_points = np.array([-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.6, 150.8, 174.1])

# Linear regression (trendline)
coeffs = np.polyfit(num_carbons, boiling_points, 1)
trendline = np.poly1d(coeffs)

# Smooth x-values for trendline
x_smooth = np.linspace(1, 10, 200)

# Figure setup
plt.figure(figsize=(9, 6))

# Scatter plot
plt.scatter(
    num_carbons,
    boiling_points,
    s=140,
    marker='h',              # hexagon marker (chemistry aesthetic)
    c=boiling_points,
    cmap='plasma',
    edgecolors='black',
    linewidth=0.8,
    alpha=0.9,
    label="Measured boiling points"
)

# Trendline
plt.plot(
    x_smooth,
    trendline(x_smooth),
    linewidth=1.8,   # thinner line
    color='black',
    alpha=0.8,
    label="Linear trend"
)

# Titles and labels
plt.title(
    "Boiling Point vs Number of Carbons\n(Linear Alkanes)",
    fontsize=16,
    fontweight='bold'
)
plt.xlabel("Number of Carbons", fontsize=13)
plt.ylabel("Boiling Point (°C)", fontsize=13)

# Grid and style
plt.grid(True, linestyle='--', alpha=0.4)

# Colorbar
cbar = plt.colorbar()
cbar.set_label("Boiling Point (°C)", fontsize=11)

# Legend
plt.legend(frameon=False, fontsize=11)

# Clean spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
