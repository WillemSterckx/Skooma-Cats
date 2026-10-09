import json
import os

import joblib
import sklearn

from mushroom_model import MAX_BATCH, parse_record, prepare, to_frame


# sagemaker calls these four functions: model_fn once at startup, the other three for every request
def model_fn(model_dir):
    with open(os.path.join(model_dir, 'model_meta.json')) as f:
        meta = json.load(f)
    if meta['versions']['sklearn'] != sklearn.__version__:
        print(f"warning: model trained with sklearn {meta['versions']['sklearn']}, serving with {sklearn.__version__}")
    return {'pipeline': joblib.load(os.path.join(model_dir, 'model.joblib')), 'meta': meta}


def input_fn(body, content_type):
    # bad input comes back as {'error': ...} so the caller gets a clear message instead of a crash
    if content_type.split(';')[0].strip().lower() != 'application/json':
        return {'error': f'content type must be application/json, got {content_type!r}'}
    try:
        data = json.loads(body)
    except ValueError:
        return {'error': 'request body is not valid JSON'}
    single = isinstance(data, dict)
    records = [data] if single else data
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_BATCH:
        return {'error': f'send one mushroom as a JSON object, or a list of 1 to {MAX_BATCH} of them'}
    return {'records': records, 'single': single}


def predict_fn(data, model):
    if 'error' in data:
        return data
    meta = model['meta']
    rows = []
    for i, record in enumerate(data['records']):
        try:
            rows.append(parse_record(record, meta['vocabulary']))
        except ValueError as e:
            return {'error': str(e) if data['single'] else f'mushroom {i}: {e}'}
    proba = model['pipeline'].predict_proba(prepare(to_frame(rows)))[:, 1].round(4)
    threshold = meta['threshold']
    # the class is decided on the same rounded number the caller sees, so the answer never contradicts itself
    results = [{'class': 'p' if p >= threshold else 'e', 'is_poisonous': int(p >= threshold),
                'probability_poisonous': float(p), 'threshold': threshold,
                'model_version': meta['model_version']} for p in proba]
    return results[0] if data['single'] else results


def output_fn(prediction, accept):
    return json.dumps(prediction), 'application/json'
