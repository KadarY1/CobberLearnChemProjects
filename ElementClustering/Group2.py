# ===============================
# Group 2 Elements Analysis (Colorful)
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# 1. Create the CSV file
# ------------------------------
csv_data = """element,symbol,atomic_number,atomic_radius_pm,first_ionization_energy_kj_mol
Beryllium,Be,4,112,900
Magnesium,Mg,12,160,738
Calcium,Ca,20,197,590
Strontium,Sr,38,215,549
Barium,Ba,56,222,503
"""

with open("group2_elements.csv", "w") as file:
    file.write(csv_data)

# ------------------------------
# 2. Load & verify the data
# ------------------------------
df = pd.read_csv("group2_elements.csv")
print(df)

# ------------------------------
# 3. Colorful scatter plot
# ------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    df["atomic_radius_pm"],
    df["first_ionization_energy_kj_mol"],
    s=120,
    c="dodgerblue",
    edgecolors="black",
    alpha=0.85
)

plt.xlabel("Atomic Radius (pm)")
plt.ylabel("First Ionization Energy (kJ/mol)")
plt.title("Group 2 Elements: Atomic Radius vs Ionization Energy")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# ------------------------------
# 4. KMeans clustering (k=2)
# ------------------------------
X = df[["atomic_radius_pm", "first_ionization_energy_kj_mol"]]

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X["atomic_radius_pm"],
    X["first_ionization_energy_kj_mol"],
    c=labels,
    cmap="plasma",
    s=140,
    edgecolors="black",
    alpha=0.9
)

# Cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=300,
    c="black",
    marker="X",
    label="Cluster Centers"
)

plt.xlabel("Atomic Radius (pm)")
plt.ylabel("First Ionization Energy (kJ/mol)")
plt.title("KMeans Clustering of Group 2 Elements (k=2)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

# ------------------------------
# 5. Interactive clustering loop
# ------------------------------
while True:
    user_input = input("Enter number of clusters (or 'q' to quit): ")

    if user_input.lower() == "q":
        print("Clustering finished.")
        break

    k = int(user_input)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X["atomic_radius_pm"],
        X["first_ionization_energy_kj_mol"],
        c=labels,
        cmap="viridis",
        s=140,
        edgecolors="black",
        alpha=0.9
    )

    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        s=300,
        c="black",
        marker="X"
    )

    plt.xlabel("Atomic Radius (pm)")
    plt.ylabel("First Ionization Energy (kJ/mol)")
    plt.title(f"KMeans Clustering (k={k})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
