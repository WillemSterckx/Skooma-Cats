# Skooma-Cats

# Members
Muhammad Azaam Ali
Willem Sterckx
Mohammed AbuQamar
Aarya Ray Chaudhuri

### 1. Mushroom dataset
Find out what is edible and what is not

### 1.1. Briefly explaining Steps
Run the notebooks in `Mushroom/` in order:
1. `1MushroomScrape.ipynb`: load the teacher's noisy dataset and explore it (class balance, missing values, noise columns, outliers, chi-square / Cramér's V, Mann-Whitney, stemless and giant groups).
2. `2MushroomClean.ipynb`: every cleaning step explained and checked with cross-validated ROC-AUC (logistic regression + random forest).
3. `2.5MushroomGraphless.ipynb`: the same cleaning in one short notebook without graphs. Both write the same `datasets/mushroom_clean.csv`.
4. `3MushroomPredict.ipynb` (split + logistic regression), `3.1MushroomPyCaret.ipynb`, `3.2MushroomRandomForest.ipynb`, `3.3MushroomGradientBoosting.ipynb` and `3.4MushroomSVM.ipynb` (SVM with an RBF kernel): one notebook per model. `4MushroomCompare.ipynb` compares them; its Part 2 adds the SVM and the SageMaker forest, with fold-by-fold and bootstrap checks. Notebooks 3.2 to 4 need scikit-learn 1.6.1, the version the saved models in `models/` were made with.
5. `3.5MushroomSageMaker.ipynb`: training, tuning and hosting the model on AWS SageMaker (runs on a SageMaker notebook instance). The guide at the top of the notebook explains which cells to run and how to put the model online. Once it is online, anyone can use the model on a web page at the API's url (`aws/page.html`).

The model code is in `Mushroom/model/` (training script + hosting code), the HTTPS front door in `Mushroom/aws/`, and the tests in `Mushroom/tests/` (`python -m pytest tests -q` from `Mushroom/`). `Mushroom/MODEL_PLAN.md` explains the plan, the API contract for the backend and every choice.

To run the model API on your own laptop (from `Mushroom/`): `python model/train.py`, then `python aws/local_api.py` → `http://127.0.0.1:8081/predict` with header `x-api-key: local-dev-key`.

### 1.2. Findings
- 5000 mushrooms, 62% edible and 38% poisonous. Poisonous is the positive class: missing a poisonous mushroom is the dangerous mistake.
- The data was damaged on purpose: random gaps (only 8 complete rows), two fake columns (cap-shape values shuffled between rows) and some flipped labels.
- Every real feature is linked to the class but weakly (Cramér's V 0.08 to 0.23). The strongest are stem-surface, cap-shape and ring-type.
- Mostly-empty columns still help: dropping spore-print-color and stem-surface cost the forest 0.032 AUC.
- Cleaning raised the random forest from 0.825 to 0.840 AUC (better in 15 of 15 folds) and logistic regression from 0.692 to 0.697.

2. City Bike dataset

Predict how many trips start at a Citi Bike station in a given hour (hourly demand per station), using all NYC trips from June 2013 to September 2026 (about 352 million trips).

2.1. Briefly explaining Steps

The data is too big for the repo (about 32 GB of zips), so everything is downloaded and built by code. Run the notebooks in CitiBike/ in order (the logic lives in CitiBike/src/, so a retraining pipeline can run the same code):

01_download.ipynb: reads the list of archives from the Citi Bike S3 bucket, downloads and unpacks them, and converts every CSV to parquet with one common set of column names (src/download.py). Values stay raw text. It can be stopped and restarted.
02_clean_eda.ipynb: cleaning and EDA on a 0.5% sample of all years: the two schemas, column names, types, duplicates, missing values, outliers (duration, location, speed), the long-term story, station IDs vs names, aggregations, and three testable hypotheses (H1 to H3). Every cleaning decision is made and explained here.
03_prepare.ipynb: the same cleaning without graphs, over all data (src/cleaning.py, in batches of 1M rows). Writes one clean parquet file per raw file.
04_baseline_model.ipynb: the demand table (src/features.py), the three-way split and four baselines (global mean, last year's average, seasonal naive, Ridge).
05_automl_flaml.ipynb: AutoML with FLAML instead of PyCaret. It searches xgboost, xgb_limitdepth, histgb, rf and extra_tree (LightGBM crashed on our Windows machines and is left out).
06_station_clusters.ipynb: unsupervised learning. KMeans groups stations by the shape of their daily demand.
07_xgboost_tuning.ipynb: XGBoost tuned by hand: complexity curves, loss functions, features, random search, final model, error analysis.
08_model_comparison.ipynb: all models side by side, validation vs test, error analysis and conclusions.

How the data is split: chronologically, never randomly. Test = the last 12 months, validation = the 12 months before, train = everything earlier (80% / 10% / 10% of the rows). Models are compared and chosen on validation only; test is reported once. Every model saves both scores in models/results.json.

The features are known in advance (station, hour, weekday, month, year, weekend, US holiday, cyclic hour, station location), so a website user only has to enter a station, a date and an hour. There are no lag features. Only the 100 busiest stations of the last 12 months are modelled, and stations are identified by name because Citi Bike changed every station ID in 2021.

To run: pip install -r requirements.txt, then run the notebooks from the CitiBike/ folder. Data and large model files are not in git (see .gitignore); running 01 and 03 rebuilds the data (it takes hours) and 04 to 07 rebuild the models. The final model is saved as models/xgb_tuned.json (plus xgb_tuned_meta.pkl) so it can be loaded without scikit-learn.

2.2. Findings
351,963,720 trips in 44 archives. The data comes in two schemas (up to January 2021 and from February 2021), mapped to one set of column names. Cleaning kept 99.73% of the trips; the biggest losses are 695,029 trips without times or coordinates and 206,421 trips faster than 40 km/h.
The system grew a lot: 8.1 million trips in 2014 and 45.8 million in 2025.
Station IDs changed in 2021 but most station names did not, so the name is the key.
Demand has strong, significant patterns by hour of day, weekday against weekend, season and year (H2 and H3 in 02), so there is something to predict.
The hardest part of the day is the evening rush (MAE about 8.8 trips per hour at 17:00 to 18:00). Summer months have larger errors than winter months because demand is higher.

Results (MAE in trips per hour, lower is better; models are ranked on validation):

Model	Validation MAE	Test MAE	Test R2
Tuned XGBoost (Poisson loss)	4.069	4.208	0.689
FLAML XGBoost	4.092	4.226	0.689
Seasonal naive (same hour last week)	4.816	5.205	0.499
Last year's average	5.076	5.447	0.514
Ridge regression	5.938	6.323	0.358
Global mean	8.359	8.935	0.000
Boosting beat bagging in the FLAML search, and the tuned XGBoost is about 15% better than the best baseline (seasonal naive, which is not usable for future hours because it needs last week's real demand). Hand tuning added only a little over FLAML; the choice of loss mattered most (Poisson 4.09 against squared error 4.18, because the target is a count).
The year matters: dropping it raised the error from 4.09 to 4.84, because demand grows every year. The station clusters from 06 did not improve the model (4.095 against 4.094), because the location already identifies the station.
The largest misses are days when the model predicts far too many trips (the ten worst days are all over-predictions). We think weather is the main missing information; this is not tested yet.
Limits: no weather, no past demand, only 100 stations, and no events or disruptions in the features.

The locally hosted model can be found at https://skoomacats.sinners.be/
