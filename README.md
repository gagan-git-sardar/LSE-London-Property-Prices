# LSE-London-Property-Prices

Predict London property prices using a LightGBM model enriched with location, crime, transport accessibility (PTAL), macro, and amenity features. The repo keeps raw data, individual EDA notebooks, and the final modeling work in a minimal, clean layout.

## Project snapshot
- Goal: estimate sale prices for London properties from multi-source features.
- Approach: feature engineering + LightGBM regression.
- Team EDA: separate notebooks for Emma, Ethan, Zhouhan; final model work in `notebooks/model/`.

## Folder map (barebones)
- `data/raw/`: original source data (crime, per‑sqm, raw property files).
- `data/external/`: reference tables (e.g., PTAL).
- `notebooks/eda/`: individual exploration notebooks.
- `notebooks/model/`: final modeling notebooks and experiments.
- `outputs/`: derived tables or presentation outputs.

## Run (quick path)
Open the modeling notebook:
- `notebooks/model/004-model.ipynb`

## Notes
Data is kept in original form under `data/` for reproducibility. The modeling notebooks document the full pipeline from cleaning to training and evaluation.
