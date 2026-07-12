import pandas as pd

def load_data():
    # Membaca dataset
    df = pd.read_csv("dataset/healthcare_dataset.csv")

    # Membersihkan Billing Amount
    if "Billing Amount" in df.columns:
        df["Billing Amount"] = (
            df["Billing Amount"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["Billing Amount"] = pd.to_numeric(
            df["Billing Amount"],
            errors="coerce"
        )

    # Membersihkan Estimated_Billing
    if "Estimated_Billing" in df.columns:
        df["Estimated_Billing"] = (
            df["Estimated_Billing"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["Estimated_Billing"] = pd.to_numeric(
            df["Estimated_Billing"],
            errors="coerce"
        )

    return df