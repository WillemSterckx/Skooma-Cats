import argparse
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)

from mushroom_model import FEATURES, build_pipeline, prepare, vocabulary

HERE = Path(__file__).resolve().parent


def number_or_name(value):
    # sagemaker sends every hyperparameter as text, max_features can be 'sqrt' or a fraction
    try:
        return float(value)
    except ValueError:
        return value


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='rf', choices=['rf', 'hgb', 'logreg'])
parser.add_argument('--n-estimators', type=int, default=200)
parser.add_argument('--min-samples-leaf', type=int, default=3)
parser.add_argument('--max-features', type=number_or_name, default='sqrt')
parser.add_argument('--max-depth', type=int, default=0)
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--max-iter', type=int, default=200)
parser.add_argument('--max-leaf-nodes', type=int, default=31)
parser.add_argument('--l2-regularization', type=float, default=0.0)
parser.add_argument('--C', type=float, default=1.0)
parser.add_argument('--target-recall', type=float, default=0.90)
parser.add_argument('--cv-repeats', type=int, default=3)
parser.add_argument('--refit-all', type=int, default=0)
# sagemaker fills these through environment variables, the defaults make the script run locally too
parser.add_argument('--train', default=os.environ.get('SM_CHANNEL_TRAIN', str(HERE.parent / 'datasets')))
parser.add_argument('--model-dir', default=os.environ.get('SM_MODEL_DIR', str(HERE.parent / 'artifacts' / 'model')))
parser.add_argument('--output-dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', str(HERE.parent / 'artifacts' / 'output')))
# parse_known_args because the tuning job also passes its own arguments, like --_tuning_objective_metric
args, _ = parser.parse_known_args()

params = {'model': args.model, 'n_estimators': args.n_estimators, 'min_samples_leaf': args.min_samples_leaf,
          'max_features': args.max_features, 'max_depth': args.max_depth or None, 'learning_rate': args.learning_rate,
          'max_iter': args.max_iter, 'max_leaf_nodes': args.max_leaf_nodes, 'l2_regularization': args.l2_regularization, 'C': args.C}
print('params', params)

raw = pd.read_csv(Path(args.train) / 'mushroom_project_dataset.csv')
X = prepare(raw)
y = (raw['class'] == 'p').astype(int)
print('rows', len(X), 'poisonous', int(y.sum()))

# same split for every model: stratified 80/20, the test part is only used for the final numbers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# cross-validated auc on the training part only, this is what the tuning job maximises
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.cv_repeats, random_state=42)
cv_aucs = cross_val_score(build_pipeline(**params), X_train, y_train, cv=cv, scoring='roc_auc')

# pick the threshold on out-of-fold predictions: the highest one that still catches target-recall of the poisonous ones
oof = cross_val_predict(build_pipeline(**params), X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
precision, recall, thresholds = precision_recall_curve(y_train, oof)
# rounded down, so rounding can never push the recall below the target
threshold = math.floor(thresholds[recall[:-1] >= args.target_recall].max() * 10000) / 10000

pipeline = build_pipeline(**params).fit(X_train, y_train)
proba = pipeline.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics = {
    'cv_auc': cv_aucs.mean(), 'cv_auc_std': cv_aucs.std(), 'threshold': threshold,
    'test_auc': roc_auc_score(y_test, proba), 'test_pr_auc': average_precision_score(y_test, proba),
    'test_recall': recall_score(y_test, pred), 'test_precision': precision_score(y_test, pred),
    'test_f1': f1_score(y_test, pred), 'test_accuracy': accuracy_score(y_test, pred),
    'train_auc': roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1]),
}
metrics = {name: round(float(value), 4) for name, value in metrics.items()}
metrics.update({'test_tp': int(tp), 'test_fp': int(fp), 'test_fn': int(fn), 'test_tn': int(tn)})

# one metric per line as name=value, the sagemaker metric regexes read these from the log
for name, value in metrics.items():
    print(f'{name}={value}')

# by default we ship the model the numbers above were measured on, --refit-all 1 retrains it on all 5000 rows (its threshold and metrics are then not measured)
if args.refit_all:
    pipeline = build_pipeline(**params).fit(X, y)

trained_at = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
meta = {
    'model_version': f'{args.model}-{trained_at}', 'trained_at': trained_at, 'params': params,
    'threshold': threshold, 'target_recall': args.target_recall, 'refit_all': bool(args.refit_all),
    'features': FEATURES, 'vocabulary': vocabulary(raw), 'metrics': metrics,
    'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
}

model_dir = Path(args.model_dir)
output_dir = Path(args.output_dir)
model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, model_dir / 'model.joblib')
(model_dir / 'model_meta.json').write_text(json.dumps(meta, indent=2))
(output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print('saved', meta['model_version'])
