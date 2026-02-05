import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# -----------------------------
# 1. Create the DataFrame
# -----------------------------

data = {
    "Molecular Weight": [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    # Molecule 11 deliberately flipped from 0 → 1
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}

molecule_names = [f"Molecule {i}" for i in range(1, 13)]
df = pd.DataFrame(data, index=molecule_names)

print("Dataset:")
print(df)
print("\n")

# -----------------------------
# 2. Split into features and target
# -----------------------------

X = df[
    ["Molecular Weight", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors"]
]
y = df["Water Solubility"]

# -----------------------------
# 3. Train Decision Tree with max depth
# -----------------------------

model = DecisionTreeClassifier(
    max_depth=3,          # <-- added constraint
    random_state=42
)
model.fit(X, y)

# -----------------------------
# 4. Feature importance (readable output)
# -----------------------------

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance (%)": (model.feature_importances_ * 100).round(1)
}).sort_values(by="Importance (%)", ascending=False)

print("Decision Tree Feature Importances (%):")
print(feature_importance_df.to_string(index=False))

# -----------------------------
# 5. Decision tree rules
# -----------------------------

print("\nDecision Tree Rules:")
print(export_text(model, feature_names=list(X.columns)))
