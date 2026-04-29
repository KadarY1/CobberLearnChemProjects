# ============================================================
# FULL MOLECULAR ML PIPELINE (FIXED VERSION)
# ============================================================

import pandas as pd
import numpy as np
import pubchempy as pcp

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# STEP 1: RELIABLE BOILING POINT DATASET (MANUAL TRUTH VALUES)
# ============================================================

molecule_data = {
    "water": 100.0,
    "ethanol": 78.37,
    "methanol": 64.7,
    "acetone": 56.05,
    "benzene": 80.1,
    "toluene": 110.6,
    "phenol": 181.7,
    "glucose": 146.0,   # decomposes but approximate
    "aspirin": 140.0,   # decomposes
    "caffeine": 178.0,  # sublimes
    "ibuprofen": 157.0,
    "acetaminophen": 420.0,
    "citric acid": 175.0,
    "formaldehyde": -19.0,
    "acetic acid": 118.1,
    "propanol": 97.2,
    "butanol": 117.7,
    "hexane": 68.7,
    "octane": 125.6,
    "chloroform": 61.2
}


# ============================================================
# STEP 2: FEATURE EXTRACTION FUNCTION
# ============================================================

def get_features(name, bp):
    try:
        compounds = pcp.get_compounds(name, "name")
        if not compounds:
            return None

        compound = compounds[0]

        smiles = compound.smiles
        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            "Molecule": name,
            "MolecularWeight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HDonors": Lipinski.NumHDonors(mol),
            "HAcceptors": Lipinski.NumHAcceptors(mol),
            "RotatableBonds": Lipinski.NumRotatableBonds(mol),
            "TPSA": Descriptors.TPSA(mol),
            "BoilingPoint": bp
        }

    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None


# ============================================================
# STEP 3: BUILD DATASET
# ============================================================

dataset = []

for mol, bp in molecule_data.items():
    print(f"Processing: {mol}")
    data = get_features(mol, bp)
    if data:
        dataset.append(data)

df = pd.DataFrame(dataset)

print("\nDataset Preview:")
print(df.head())


# ============================================================
# STEP 4: CLEAN DATA (NO IMPUTATION ON TARGET)
# ============================================================

df = df.dropna()

print("\nAfter Cleaning:")
print(df.isnull().sum())


# ============================================================
# STEP 5: SAVE DATASET
# ============================================================

df.to_csv("my_project_dataset.csv", index=False)
print("\nSaved: my_project_dataset.csv")


# ============================================================
# STEP 6: LOAD DATASET
# ============================================================

data = pd.read_csv("my_project_dataset.csv")


# ============================================================
# STEP 7: SPLIT FEATURES AND TARGET
# ============================================================

X = data[
    [
        "MolecularWeight",
        "LogP",
        "HDonors",
        "HAcceptors",
        "RotatableBonds",
        "TPSA"
    ]
]

y = data["BoilingPoint"]


# ============================================================
# STEP 8: TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# STEP 9: TRAIN MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# STEP 10: EVALUATE MODEL
# ============================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL RESULTS =====")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")


# ============================================================
# STEP 11: FEATURE IMPORTANCE (OPTIONAL BUT GOOD FOR REPORT)
# ============================================================

importances = model.feature_importances_

for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.3f}")