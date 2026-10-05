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
4. `3MushroomPredict.ipynb` and `4MushroomCompare.ipynb`: models and comparison (in progress).

### 1.2. Findings
- 5000 mushrooms, 62% edible and 38% poisonous. Poisonous is the positive class: missing a poisonous mushroom is the dangerous mistake.
- The data was damaged on purpose: random gaps (only 8 complete rows), two fake columns (cap-shape values shuffled between rows) and some flipped labels.
- Every real feature is linked to the class but weakly (Cramér's V 0.08 to 0.23). The strongest are stem-surface, cap-shape and ring-type.
- Mostly-empty columns still help: dropping spore-print-color and stem-surface cost the forest 0.032 AUC.
- Cleaning raised the random forest from 0.825 to 0.840 AUC (better in 15 of 15 folds) and logistic regression from 0.692 to 0.697.

### 2. City Bike dataset

### 2.1. Briefly explaining Steps

### 2.2. Findings