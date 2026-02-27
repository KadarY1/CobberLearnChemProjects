import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate noisy dataset
# -----------------------------
np.random.seed(42)

x = np.linspace(0, 10, 20)
y_true = 2 * x + 5
noise = np.random.normal(0, 2, size=len(x))
y = y_true + noise

# -----------------------------
# 2. Define MSE function
# -----------------------------
def calculate_mse(slope, intercept, x, y):
    y_pred = slope * x + intercept
    return np.mean((y - y_pred) ** 2)

# -----------------------------
# 3. Create grid of parameters
# -----------------------------
slope_values = np.linspace(0, 4, 100)
intercept_values = np.linspace(0, 10, 100)

SLOPE, INTERCEPT = np.meshgrid(slope_values, intercept_values)
MSE = np.zeros_like(SLOPE)

# -----------------------------
# 4. Compute MSE at each grid point
# -----------------------------
for i in range(SLOPE.shape[0]):
    for j in range(SLOPE.shape[1]):
        MSE[i, j] = calculate_mse(
            SLOPE[i, j],
            INTERCEPT[i, j],
            x,
            y
        )

# -----------------------------
# 5. Plot loss landscape
# -----------------------------
plt.figure(figsize=(8, 6))
plt.contourf(
    SLOPE,
    INTERCEPT,
    MSE,
    levels=50,
    cmap="viridis_r"  # yellow = low MSE, purple = high MSE
)

plt.xlabel("Slope")
plt.ylabel("Intercept")
plt.title("Loss Landscape (Mean Squared Error)")
plt.colorbar(label="MSE")
plt.show()