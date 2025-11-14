import pandas as pd
import glob
import os

path = "."  # same folder as this script

# Check what files are found
files = glob.glob("*.csv")
print("Found CSV files:", len(files))
print(files)

for file in files:
    try:
        print(f"Processing: {file}")
        df = pd.read_csv(file)

        # Clean by year
        df_clean = df[(df["year"] >= 2015) & (df["year"] <= 2024)]

        # Save back
        df_clean.to_csv(file, index=False)

        print(f"  → Saved {len(df_clean)} rows")

    except Exception as e:
        print(f"Error with {file}: {e}")
