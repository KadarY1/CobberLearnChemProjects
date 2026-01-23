from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# -----------------------------
#   Input SMILES for Ibuprofen
# -----------------------------
smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError("Invalid SMILES string")

# -----------------------------
#   Descriptor Calculations
# -----------------------------
exact_mw = Descriptors.ExactMolWt(mol)
hbd = rdMolDescriptors.CalcNumHBD(mol)
tpsa = rdMolDescriptors.CalcTPSA(mol)

# Additional descriptors:
logp = Descriptors.MolLogP(mol)
rot_bonds = Descriptors.NumRotatableBonds(mol)

# -----------------------------
#   Pretty Output Formatting
# -----------------------------
print("\n" + "="*46)
print("        🧪 Ibuprofen Molecular Properties")
print("="*46)

print(f"{'SMILES:':25} {smiles}")
print("-"*46)
print(f"{'Exact Molecular Weight:':25} {exact_mw:10.4f}")
print(f"{'H-Bond Donors:':25} {hbd:10}")
print(f"{'TPSA:':25} {tpsa:10.4f}")
print(f"{'LogP:':25} {logp:10.4f}")
print(f"{'Rotatable Bonds:':25} {rot_bonds:10}")

print("="*46 + "\n")
