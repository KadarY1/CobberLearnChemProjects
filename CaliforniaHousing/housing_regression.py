"""
California Housing Price Prediction Project
--------------------------------------------
This script:
1. Loads and explores the California Housing dataset
2. Trains multiple regression models
3. Evaluates models using R² and cross-validation
4. Creates and saves visualizations (histograms + heatmap)
5. Uses the best model for predictions
6. Allows user to input multiple predictions interactively
"""

# -------------------------------
# Imports
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# -------------------------------
# Step 2: Explore Data
# -------------------------------
print("\n--- Dataset Overview ---")
print("\nFirst 5 rows:\n", df.head())
print("\nBasic Statistics:\n", df.describe())

# -------------------------------
# Step 3: Visualizations
# -------------------------------

# Histograms for each feature
df.hist(figsize=(12, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("\nSaved visualizations: histograms.png, correlation_heatmap.png")

# -------------------------------
# Step 4: Prepare Data
# -------------------------------
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 5: Train Models
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR()
}

results = {}

print("\n--- Model Performance ---")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mean = np.mean(cv_scores)

    results[name] = {
        "model": model,
        "r2": r2,
        "cv": cv_mean
    }

    print(f"{name}:")
    print(f"  Test R² = {r2:.4f}")
    print(f"  CV R²   = {cv_mean:.4f}\n")

# -------------------------------
# Step 6: Find Best Model
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]["cv"])
best_model = results[best_model_name]["model"]

print(f"Best Model: {best_model_name}")

# -------------------------------
# Step 7: Predicted vs Actual Plot
# -------------------------------
preds = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title(f"Predicted vs Actual ({best_model_name})")
plt.savefig("pred_vs_actual.png")
plt.close()

print("Saved visualization: pred_vs_actual.png")

# -------------------------------
# Step 8: Predict New Data
# -------------------------------
print("\n--- Prediction Tool ---")
print("Enter values for a new house (or type 'q' to quit)\n")

feature_names = housing.feature_names

while True:
    user_input = input("Make a prediction? (y/n): ").lower()

    if user_input == 'n':
        print("Exiting prediction tool.")
        break

    elif user_input == 'y':
        values = []

        for feature in feature_names:
            val = input(f"Enter {feature}: ")

            if val.lower() == 'q':
                print("Exiting prediction tool.")
                exit()

            values.append(float(val))

        # Convert input into array
        new_data = np.array(values).reshape(1, -1)

        prediction = best_model.predict(new_data)

        print(f"\nPredicted House Price: ${prediction[0]*100000:.2f}\n")

    else:
        print("Please enter 'y' or 'n'.")

# -------------------------------
# Notes
# -------------------------------
"""
- Code is organized into clear sections
- Comments explain each step
- Output is clean and readable
- Visualizations are saved as PNG files
- Cross-validation improves reliability
- Interactive loop allows multiple predictions
"""