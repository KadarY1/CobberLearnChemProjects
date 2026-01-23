import pubchempy as pcp
import ssl
import certifi
import pubchempy as pcp

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

compounds = pcp.get_compounds("theobromine", "name")
print(compounds[0].molecular_formula)

def fetch_theobromine_data():
    # Search compound by name
    compounds = pcp.get_compounds("theobromine", "name")

    if not compounds:
        raise ValueError("Theobromine not found in PubChem.")

    compound = compounds[0]

    data = {
        "CID": compound.cid,
        "Molecular Formula": compound.molecular_formula,
        "Molecular Weight": compound.molecular_weight,
        "IUPAC Name": compound.iupac_name,
        "Canonical SMILES": compound.canonical_smiles,
        "InChI": compound.inchi,
        "InChI Key": compound.inchikey,
        "XLogP": compound.xlogp,
        "Synonyms": compound.synonyms[:10] if compound.synonyms else []
    }

    return data


if __name__ == "__main__":
    theobromine_data = fetch_theobromine_data()

    for key, value in theobromine_data.items():
        print(f"{key}: {value}")
