import json
import re
import subprocess
import sys

import pandas as pd
import pytest

from conftest import MUSHROOM, RAW_CSV, run_training
from mushroom_model import FEATURES, prepare


def test_training_job_writes_model_and_metrics(trained):
    assert (trained['dir'] / 'model.joblib').exists()
    meta = json.loads((trained['dir'] / 'model_meta.json').read_text())
    metrics = json.loads((trained['output'] / 'metrics.json').read_text())
    assert meta['features'] == FEATURES
    assert meta['params']['n_estimators'] == 40 and meta['params']['max_features'] == 0.5
    assert 0 < meta['threshold'] < 1 and meta['threshold'] == round(meta['threshold'], 4)
    assert meta['refit_all'] is False
    assert 0.7 < metrics['cv_auc'] < 1 and 0.7 < metrics['test_auc'] < 1
    assert metrics['test_tp'] + metrics['test_fp'] + metrics['test_fn'] + metrics['test_tn'] == 1000


def test_sagemaker_metric_regexes_find_every_metric(trained):
    # the tuning job reads its objective from the log with these regexes, if one breaks the job has nothing to tune on
    definitions = json.loads((MUSHROOM / 'aws' / 'metric_definitions.json').read_text())
    metrics = json.loads((trained['output'] / 'metrics.json').read_text())
    for d in definitions:
        found = [float(m) for m in re.findall(d['Regex'], trained['log'])]
        assert found == [metrics[d['Name']]], d['Name']


def test_pickle_loads_without_our_code(trained, tmp_path):
    # the saved pipeline is plain sklearn, so it loads in a python that can't see our model/ folder
    prepare(pd.read_csv(RAW_CSV)).head(50).to_csv(tmp_path / 'rows.csv', index=False)
    script = ("import joblib, pandas as pd, sys; m = joblib.load(sys.argv[1]); "
              "print(','.join(f'{p:.6f}' for p in m.predict_proba(pd.read_csv(sys.argv[2]))[:, 1]))")
    out = subprocess.run([sys.executable, '-P', '-c', script, str(trained['dir'] / 'model.joblib'), str(tmp_path / 'rows.csv')],
                         cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert len(out.stdout.strip().split(',')) == 50


def test_tuning_job_arguments_are_accepted(tmp_path):
    # a tuning job adds its own hyperparameters to every training job, the script must not stop on them
    log = run_training(tmp_path, '--model', 'rf', '--n-estimators', '20', '--cv-repeats', '1', '--_tuning_objective_metric', 'cv_auc')
    assert 'cv_auc=' in log


@pytest.mark.parametrize('extra', [
    ['--model', 'hgb', '--max-iter', '50', '--learning-rate', '0.05', '--min-samples-leaf', '20'],
    ['--model', 'logreg', '--C', '0.5'],
])
def test_other_model_families_train(tmp_path, extra):
    log = run_training(tmp_path, *extra, '--cv-repeats', '1', '--refit-all', '1')
    assert 'cv_auc=' in log
    meta = json.loads((tmp_path / 'model' / 'model_meta.json').read_text())
    assert meta['params']['model'] == extra[1] and meta['refit_all'] is True
