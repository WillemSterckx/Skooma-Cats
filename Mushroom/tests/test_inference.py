import json

import numpy as np
import pandas as pd
import pytest

import inference
from conftest import EXAMPLES, RAW_CSV
from mushroom_model import FEATURES, prepare


def ask(model, body, content_type='application/json'):
    # one request through the same four functions sagemaker calls
    text, mime = inference.output_fn(inference.predict_fn(inference.input_fn(body, content_type), model), 'application/json')
    assert mime == 'application/json'
    return json.loads(text)


def example(name):
    return (EXAMPLES / name).read_text()


def test_full_example(model):
    answer = ask(model, example('full.json'))
    assert set(answer) == {'class', 'is_poisonous', 'probability_poisonous', 'threshold', 'model_version'}
    assert answer['class'] in ('e', 'p') and 0 <= answer['probability_poisonous'] <= 1
    assert answer['is_poisonous'] == int(answer['probability_poisonous'] >= answer['threshold'])
    assert answer['class'] == ('p' if answer['is_poisonous'] else 'e')


def test_form_example_with_text_numbers_blanks_and_noise(model):
    form = ask(model, example('from_form.json'))
    typed = ask(model, json.dumps({'stem-height': 6.42, 'stem-width': 13.51, 'habitat': 'd', 'season': 'u', 'ring-type': 'g', 'cap-shape': 'x'}))
    assert form['probability_poisonous'] == typed['probability_poisonous']


def test_batch_example_returns_a_list(model):
    answer = ask(model, example('batch.json'))
    assert isinstance(answer, list) and len(answer) == 2


def test_invalid_example_says_what_is_wrong(model):
    assert 'cap-diameter' in ask(model, example('invalid.json'))['error']


def test_empty_mushroom_still_gets_a_prediction(model):
    assert ask(model, '{}')['class'] in ('e', 'p')


def test_missing_none_blank_and_uppercase_mean_the_same(model):
    base = ask(model, json.dumps({'habitat': 'd', 'gill-color': 'w'}))
    for variant in [{'habitat': 'D', 'gill-color': ' w ', 'season': 'missing'}, {'habitat': 'd', 'gill-color': 'w', 'season': None, 'cap-diameter': ''}]:
        assert ask(model, json.dumps(variant))['probability_poisonous'] == base['probability_poisonous']


def test_api_path_gives_the_same_numbers_as_the_pipeline(model):
    # 200 real rows sent as JSON must score exactly like the pipeline scores the dataframe directly
    raw = pd.read_csv(RAW_CSV).head(200)
    records = [{k: (None if pd.isna(v) else v) for k, v in row.items() if k != 'class'} for row in raw.to_dict('records')]
    for i in range(0, 200, 100):
        answers = ask(model, json.dumps(records[i:i + 100]))
        direct = model['pipeline'].predict_proba(prepare(raw.iloc[i:i + 100]))[:, 1]
        assert np.allclose([a['probability_poisonous'] for a in answers], direct.round(4), atol=1e-4)


@pytest.mark.parametrize('body, expected', [
    ('{"cap_diameter": 5}', 'unknown field'),
    ('{"habitat": "zz"}', 'habitat must be one of'),
    ('{"stem-height": -1}', '>= 0'),
    ('{"stem-height": true}', 'stem-height'),
    ('{"stem-height": "nan"}', '>= 0'),
    ('{"cap-diameter": 1' + '0' * 400 + '}', 'cap-diameter'),
    ('{"cap-diameter": [1]}', 'cap-diameter'),
    ('not json', 'not valid JSON'),
    ('[]', 'list of 1 to 100'),
    ('"text"', 'list of 1 to 100'),
    ('[{"habitat": "d"}, 5]', 'mushroom 1'),
    (json.dumps([{}] * 101), 'list of 1 to 100'),
])
def test_bad_input_returns_an_error_message(model, body, expected):
    answer = ask(model, body)
    assert set(answer) == {'error'} and expected in answer['error']


def test_wrong_content_type_is_rejected(model):
    assert 'application/json' in ask(model, 'a,b', 'text/csv')['error']


def test_meta_lists_every_feature_and_code(model):
    meta = model['meta']
    assert meta['features'] == FEATURES
    assert meta['vocabulary']['season'] == ['a', 's', 'u', 'w']
