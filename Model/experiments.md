## Experiment 001

**Base Modele**
- Uses Optuna hyperparameter tuning n = 200
- target price no transformation
- 12 features
Uses random k-fold cross-validation

**Result**
- Systematic overeprediction of low prices and underprediction of high prices
- RMSE explodes at first APE decile because of large overprediction of low property prices
- Observed heteroskedacity as we approach higher price ranges

## Experiment 002

**Change**
- Target = log(price)
- Loss = Huber(alpha=0.9)
- Include borough feature (coarse geospatial)
- include postcode feature (fine geospatial)
- Include lat long feature (finest geospatial)
- Cross-validation is no longer used, but instead a time-based split!! Not interpolation, but extrapolation! This distinction is super important and improved model performance drastically

**Motivation**
- Reduce heteroskedasticity
- Penalize large errors less aggressively
- Explain model more fully (fine geospatial features turned out to be really good at increasing predictions of high property prices and lowering predictions of low property prices)

**Hyperparams**
{'n_estimators': 1156, 'learning_rate': 0.10763245775290799, 'num_leaves': 143, 'max_depth': 17, 'min_child_samples': 20, 'subsample': 0.8531853409366144, 'colsample_bytree': 0.5670837178989158}

**Result**
- Reduced variance in mid-range
- Persistent overprediction at low end (SPE << 0)
- High-end bias slightly better but still exists

RMSE: 123305.86
Mean price: 592617.05
RMSE as percentage of mean price: 20.81%
MAE: 78629.01
MAE as percentage of mean price: 13.27%
R-squared: 0.85

Within  5%:  28.1% (9427/33598 properties)
Within 10%:  52.0% (17476/33598 properties)
Within 15%:  69.2% (23247/33598 properties)
Within 20%:  80.7% (27127/33598 properties)

**Decision**
- Segmentation across different property tiers

## Experiment 003

**Results**
- This is an interim model used to decide the borders for segmentation, and play around with dropping values that would maximise the decrease in APE

Primary split (~£425k): SPE mean flips from negative to positive between (420k, 428k] and (428k, 437.5k]. This is the cleanest boundary between systematic over‑prediction vs under‑prediction. Biggest expected gain.
Low-end segment (≤£225k or ≤£250k): The lowest bins have very large negative SPE and huge variance (e.g., −0.75 mean, std 0.90). A dedicated low‑price model should reduce the extreme over‑prediction and variance.
High-end segment (≥£1.1m or ≥£1.2m): SPE mean rises notably from ~0.04–0.06 to ~0.08–0.17, and the bias grows steadily. A separate high‑end model should correct under‑prediction.

Practical segmentation options (ranked by expected gain vs complexity):
Two segments (best ROI):
<= ~£425k and > ~£425k.
Three segments (likely best overall):
<= £250k, £250k–£1.1m, > £1.1m
(or use £225k and £1.2m as alternatives).
Four segments (if you can afford more models):
<= £225k, £225k–£425k, £425k–£1.1m, > £1.1m.
If you want, I can add a quick cell to fit these splits and compare RMSE/MAE so you can choose based on actual lift.

## Experiment 004

**Change**
- Three segmented models based on train data evaluation:
<= £250k, £250k–£1.1m, > £1.1m

- Linear calibration to fix underestimation of high prices and overestimation of low prices

**Results**

- Still has systematic overprediction of low end and underprediction of high end within each segmented model

RMSE: 96563.17
Mean price: 592617.05
RMSE as percentage of mean price: 16.29%
MAE: 65554.66
MAE as percentage of mean price: 11.06%
R-squared: 0.91

Within  5%:  30.9% (10384/33598 properties)
Within 10%:  56.6% (19017/33598 properties)
Within 15%:  74.6% (25079/33598 properties)
Within 20%:  85.3% (28670/33598 properties)

**Results (Including Linear Calibration)**
RMSE: 95912.48
Mean price: 592617.05
RMSE as percentage of mean price: 16.18%
MAE: 64826.34
MAE as percentage of mean price: 10.94%
R-squared: 0.91

Within  5%:  31.6% (10625/33598 properties)
Within 10%:  57.3% (19242/33598 properties)
Within 15%:  74.8% (25124/33598 properties)
Within 20%:  85.2% (28642/33598 properties)


**Hyperparams**
Best Parameters for low segment : 
{'n_estimators': 1195, 'learning_rate': 0.03207664578812468, 'num_leaves': 84, 'max_depth': 13, 'min_child_samples': 5, 'subsample': 0.7889533873638548, 'colsample_bytree': 0.5663454662129782}

Best Parameters for mid segment : 
{'n_estimators': 1494, 'learning_rate': 0.0591310714508011, 'num_leaves': 143, 'max_depth': 19, 'min_child_samples': 18, 'subsample': 0.9327023224082089, 'colsample_bytree': 0.5610102961609175}

Best Parameters for high segment : 
{'n_estimators': 1011, 'learning_rate': 0.010555203732718746, 'num_leaves': 130, 'max_depth': 18, 'min_child_samples': 12, 'subsample': 0.8753968850598899, 'colsample_bytree': 0.5003753045491645}