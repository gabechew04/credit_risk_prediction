from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "datasets"
LOANS_DIR = DATA_DIR / "loan_cleaned.csv"
CUSTOMERS_DIR = DATA_DIR / "customer.csv"
OUTPUT_DIR = DATA_DIR / "master.csv"

loans = pd.read_csv(LOANS_DIR)
customers = pd.read_csv(CUSTOMERS_DIR)

merged_df = pd.merge(customers, loans, how="inner", on="customer_id")
print(f"Sucessfully merged datasets in \
      \n{LOANS_DIR} \
      \nand \
      \n{CUSTOMERS_DIR}")
merged_df.to_csv(OUTPUT_DIR, index=False)
print(f"Wrote merged loan csv to {OUTPUT_DIR}")






