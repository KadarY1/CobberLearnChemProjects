import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Given data
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Residuals
residuals = predicted - actual

# --- Column Display (DataFrame) ---
df = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted,
    "Residual": residuals
})

print(df)

# --- Metrics using scikit-learn ---
mae_sklearn = mean_absolute_error(actual, predicted)
mse_sklearn = mean_squared_error(actual, predicted)
r2_sklearn = r2_score(actual, predicted)

print("\nScikit-learn MAE:", mae_sklearn)
print("Scikit-learn MSE:", mse_sklearn)
print("Scikit-learn R^2:", r2_sklearn)

# --- Predicted vs Actual Scatter Plot ---
plt.figure(figsize=(6, 5))
plt.scatter(actual, predicted, color='darkorange', s=120)  # bigger points, new color
plt.plot(actual, actual, color='blue', linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual")
plt.grid(True)
plt.savefig("predicted_vs_actual.png")
plt.close()

# --- Residual Plot ---
plt.figure(figsize=(6, 5))
plt.scatter(actual, residuals, color='green', s=120)  # bigger points, new color
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.savefig("residual_plot.png")
plt.close()
