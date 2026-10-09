import numpy as np
import pandas as pd

from conftest import CLEAN_CSV, RAW_CSV
from mushroom_model import CLEAN_CAT, CLEAN_NUM, prepare


def test_prepare_matches_the_cleaning_notebooks():
    # prepare() must do exactly what notebook 2 / 2.5 do, except the imputer which now lives in the pipeline
    prepared = prepare(pd.read_csv(RAW_CSV))
    clean = pd.read_csv(CLEAN_CSV)
    assert list(prepared.columns) == CLEAN_NUM + CLEAN_CAT
    assert list(clean.columns) == CLEAN_NUM + CLEAN_CAT + ['is_poisonous']
    assert (prepared[CLEAN_CAT].astype(str).values == clean[CLEAN_CAT].astype(str).values).all()
    known = prepared[CLEAN_NUM].notna().to_numpy()
    assert np.allclose(prepared[CLEAN_NUM].to_numpy()[known], clean[CLEAN_NUM].to_numpy()[known], atol=0.005)


def test_only_measurements_are_left_empty():
    prepared = prepare(pd.read_csv(RAW_CSV))
    assert prepared[CLEAN_CAT].notna().all().all()
    assert prepared[CLEAN_NUM].isna().sum().sum() > 3000


def test_stemless_rule_fills_both_ways():
    raw = pd.DataFrame({'cap-diameter': [3.0, 3.0], 'stem-height': [0.0, np.nan], 'stem-width': [np.nan, np.nan],
                        'stem-surface': [np.nan, 'f'], 'jumbled_noise_0': ['x', 'x']})
    for col in ['spore-print-color', 'gill-color', 'habitat', 'season', 'ring-type', 'cap-shape']:
        raw[col] = np.nan
    prepared = prepare(raw)
    assert prepared['stem_surface'].tolist() == ['f', 'f']
    assert (prepared[['stem_height_cm', 'stem_width_mm']] == 0).all().all()
    assert 'jumbled_noise_0' not in prepared
