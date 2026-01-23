import pubchempy as pcp


def fetch_compound_summary(compound_name: str) -> dict:
    """
    Fetch molecular data for a compound from PubChem.

    Parameters
    ----------
    compound_name : str
        Common or IUPAC name of the compound.

    Returns
    -------
    dict
        Dictionary containing key molecular properties.
    """
    compounds = pcp.get_compounds(compound_name, "name")

    if not compounds:
        raise ValueError(f"No PubChem entry found for '{compound_name}'.")

    c = compounds[0]

    return {
        "Name": compound_name.title(),
        "CID": c.cid,
        "Molecular Formula": c.molecular_formula,
        "Molecular Weight": f"{c.molecular_weight:.2f} g/mol",
        "SMILES": c.canonical_smiles,
    }


def print_compound_summary(data: dict) -> None:
    """Pretty-print compound data."""
    print("\n" + "═" * 50)
    print(f"🧬  Compound Overview: {data['Name']}")
    print("═" * 50)

    for key, value in data.items():
        if key != "Name":
            print(f"🔹 {key:<20}: {value}")

    print("═" * 50 + "\n")


if __name__ == "__main__":
    compound_name = "theobromine"  # change this to any compound you want

    try:
        compound_data = fetch_compound_summary(compound_name)
        print_compound_summary(compound_data)
    except Exception as e:
        print("❌ Error:", e)
