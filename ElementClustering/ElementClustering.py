# =========================
# Group 1 Elements Analysis
# =========================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------
# 1. Create the CSV file
# -------------------------
csv_data = """element,symbol,atomic_number,atomic_radius_pm,first_ionization_energy_kj_mol
Lithium,Li,3,152,520
Sodium,Na,11,186,496
Potassium,K,19,227,419
Rubidium,Rb,37,248,403
Cesium,Cs,55,265,376
"""

with open("group1_elements.csv", "w") as file:
    file.write(csv_data)

# -------------------------
# 2. Load & verify the data
# -------------------------
df = pd.read_csv("group1_elements.csv")

print("Data loaded successfully:\n")
print(df)
print("\nColumns:", df.columns)
print("\nData types:\n", df.dtypes)

# -------------------------
# 3. Scatter plot
# -------------------------
plt.figure()
plt.scatter(
    df["atomic_radius_pm"],
    df["first_ionization_energy_kj_mol"],
    s=80,
    alpha=0.8
)

plt.xlabel("Atomic Radius (pm)")
plt.ylabel("First Ionization Energy (kJ/mol)")
plt.title("Group 1 Elements: Atomic Radius vs Ionization Energy")
plt.show()

# -------------------------
# 4. Initial KMeans (k=2)
# -------------------------
X = df[["atomic_radius_pm", "first_ionization_energy_kj_mol"]]

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

plt.figure()
plt.scatter(
    X["atomic_radius_pm"],
    X["first_ionization_energy_kj_mol"],
    c=labels,
    s=80
)

centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=200,
    c="black",
    marker="x"
)

plt.xlabel("Atomic Radius (pm)")
plt.ylabel("First Ionization Energy (kJ/mol)")
plt.title("KMeans Clustering of Group 1 Elements (k=2)")
plt.show()

# -------------------------
# 5. Interactive clustering loop
# -------------------------
while True:
    user_input = input("Enter number of clusters (or 'q' to quit): ")

    if user_input.lower() == "q":
        print("Clustering finished.")
        break

    k = int(user_input)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    plt.figure()
    plt.scatter(
        X["atomic_radius_pm"],
        X["first_ionization_energy_kj_mol"],
        c=labels,
        s=80
    )

    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        s=200,
        c="black",
        marker="x"
    )

    plt.xlabel("Atomic Radius (pm)")
    plt.ylabel("First Ionization Energy (kJ/mol)")
    plt.title(f"KMeans Clustering (k={k})")
    plt.show()
