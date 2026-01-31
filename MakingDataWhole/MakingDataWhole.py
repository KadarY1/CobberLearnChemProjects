import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load data
# -------------------------
df = sns.load_dataset("titanic")
print(f"\nDataset loaded — Rows: {df.shape[0]}  Columns: {df.shape[1]}")

# -------------------------
# Correlation matrix
# -------------------------
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().round(3)

print("\n=== Correlation Matrix (Rounded) ===")
print(corr_matrix)

# Clearer heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, interpolation="nearest")
plt.title("Titanic Feature Correlation Matrix")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=60)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# -------------------------
# Correlation with Age (sorted + formatted)
# -------------------------
age_corr = corr_matrix["age"].drop("age")
age_corr_sorted = age_corr.reindex(age_corr.abs().sort_values(ascending=False).index)

print("\n=== Features Most Correlated With Age ===")
print(age_corr_sorted)

print("\nTop 2 strongest correlations with Age:")
print(age_corr_sorted.head(2))

# -------------------------
# KNN Imputation
# -------------------------
features = numeric_df.columns.tolist()
knn_data = numeric_df[features]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(knn_data)

imputer = KNNImputer(n_neighbors=5)
imputed_scaled = imputer.fit_transform(scaled_data)

imputed_data = pd.DataFrame(
    scaler.inverse_transform(imputed_scaled),
    columns=features
)

# -------------------------
# Evaluate predictions
# -------------------------
known_mask = df["age"].notna()
actual_ages = knn_data.loc[known_mask, "age"]
predicted_ages = imputed_data.loc[known_mask, "age"]

mae = mean_absolute_error(actual_ages, predicted_ages)

print(f"\nKNN Age Prediction MAE: {mae:.3f}")

# -------------------------
# Improved prediction plot
# -------------------------
plt.figure(figsize=(8, 6))
plt.scatter(actual_ages, predicted_ages, alpha=0.5)
plt.plot([0, 80], [0, 80])  # reference perfect-fit line
plt.xlabel("Actual Age")
plt.ylabel("KNN Predicted Age")
plt.title("Actual vs KNN Predicted Ages")
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_age_predictions.png")
plt.close()

# -------------------------
# Impute missing ages
# -------------------------
avg_before = df["age"].mean()

df_knn = df.copy()
df_knn["age"] = imputed_data["age"]

avg_after = df_knn["age"].mean()

print("\n=== Age Imputation Summary ===")
print(f"Average Age Before KNN: {avg_before:.2f}")
print(f"Average Age After KNN:  {avg_after:.2f}")
print("Missing Ages After KNN:", df_knn["age"].isna().sum())
