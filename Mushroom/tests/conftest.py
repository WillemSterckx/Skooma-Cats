import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MUSHROOM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MUSHROOM / 'model'))
sys.path.insert(0, str(MUSHROOM / 'aws'))

RAW_CSV = MUSHROOM / 'datasets' / 'mushroom_project_dataset.csv'
CLEAN_CSV = MUSHROOM / 'datasets' / 'mushroom_clean.csv'
EXAMPLES = MUSHROOM / 'aws' / 'examples'


def run_training(tmp, *extra):
    # runs train.py the way a sagemaker training job does: data and output folders come from SM_* variables
    train_dir = tmp / 'input' / 'data' / 'train'
    train_dir.mkdir(parents=True)
    shutil.copy(RAW_CSV, train_dir)
    env = {**os.environ, 'SM_CHANNEL_TRAIN': str(train_dir), 'SM_MODEL_DIR': str(tmp / 'model'), 'SM_OUTPUT_DATA_DIR': str(tmp / 'output')}
    result = subprocess.run([sys.executable, str(MUSHROOM / 'model' / 'train.py'), *extra],
                            cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(scope='session')
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('job')
    log = run_training(tmp, '--model', 'rf', '--n-estimators', '40', '--max-features', '0.5', '--cv-repeats', '1')
    return {'dir': tmp / 'model', 'output': tmp / 'output', 'log': log}


@pytest.fixture(scope='session')
def model(trained):
    import inference
    return inference.model_fn(str(trained['dir']))
