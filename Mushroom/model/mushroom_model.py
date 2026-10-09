import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# raw names (what the API and the raw csv use) -> clean names with units (notebook 2)
NUM_COLS = {'cap-diameter': 'cap_diameter_cm', 'stem-height': 'stem_height_cm', 'stem-width': 'stem_width_mm'}
CAT_COLS = ['spore-print-color', 'gill-color', 'habitat', 'season', 'ring-type', 'cap-shape', 'stem-surface']
NOISE_COLS = ['jumbled_noise_0', 'jumbled_noise_1']
FEATURES = list(NUM_COLS) + CAT_COLS
CLEAN_NUM = list(NUM_COLS.values())
CLEAN_CAT = [c.replace('-', '_') for c in CAT_COLS]
MAX_BATCH = 100


def prepare(raw):
    # the cleaning steps that learn nothing from the data, same as 2.5MushroomGraphless steps 1, 3 and 4
    df = raw[FEATURES].rename(columns=NUM_COLS)
    df.columns = df.columns.str.replace('-', '_')
    df[CLEAN_NUM] = df[CLEAN_NUM].astype(float)
    df[CLEAN_CAT] = df[CLEAN_CAT].astype(object)
    zero_stem = (df['stem_height_cm'] == 0) | (df['stem_width_mm'] == 0)
    df.loc[zero_stem, 'stem_surface'] = 'f'
    df.loc[df['stem_surface'] == 'f', ['stem_height_cm', 'stem_width_mm']] = 0
    df[CLEAN_CAT] = df[CLEAN_CAT].fillna('missing')
    return df


def vocabulary(raw):
    return {col: sorted(raw[col].dropna().unique().tolist()) for col in CAT_COLS}


def is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == '')


def parse_record(record, vocab):
    # turns one JSON object from the API into a raw row, or raises ValueError with a message for the user
    if not isinstance(record, dict):
        raise ValueError('each mushroom must be a JSON object')
    unknown = sorted(set(record) - set(FEATURES) - set(NOISE_COLS))
    if unknown:
        raise ValueError(f'unknown field(s) {unknown}, allowed fields are {FEATURES}')
    row = {}
    for col in NUM_COLS:
        value = record.get(col)
        if is_blank(value):
            row[col] = math.nan
            continue
        if isinstance(value, bool):
            raise ValueError(f'{col} must be a number in {NUM_COLS[col][-2:]} or empty, got {value!r}')
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{col} must be a number in {NUM_COLS[col][-2:]} or empty, got {value!r}')
        if not math.isfinite(number) or number < 0:
            raise ValueError(f'{col} must be a number >= 0, got {value!r}')
        row[col] = number
    for col in CAT_COLS:
        value = record.get(col)
        code = '' if is_blank(value) else str(value).strip().lower()
        if code in ('', 'missing'):
            row[col] = math.nan
        elif code in vocab[col]:
            row[col] = code
        else:
            raise ValueError(f'{col} must be one of {vocab[col]} or empty, got {value!r}')
    return row


def build_pipeline(model='rf', n_estimators=200, min_samples_leaf=3, max_features='sqrt', max_depth=None,
                   learning_rate=0.1, max_iter=200, max_leaf_nodes=31, l2_regularization=0.0, C=1.0):
    # the imputer sits inside the pipeline so it only ever learns from the training rows
    num_steps = [FunctionTransformer(np.log1p, feature_names_out='one-to-one'), IterativeImputer(max_iter=25, random_state=42)]
    if model == 'logreg':
        num_steps.append(StandardScaler())
    prep = ColumnTransformer([
        ('num', make_pipeline(*num_steps), CLEAN_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CLEAN_CAT),
    ])
    if model == 'rf':
        clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                     max_depth=max_depth, random_state=42, n_jobs=-1)
    elif model == 'hgb':
        clf = HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                                             min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization, random_state=42)
    elif model == 'logreg':
        clf = LogisticRegression(C=C, max_iter=2000)
    else:
        raise ValueError(f'unknown model {model!r}, use rf, hgb or logreg')
    return make_pipeline(prep, clf)


def to_frame(rows):
    return pd.DataFrame(rows, columns=FEATURES)
