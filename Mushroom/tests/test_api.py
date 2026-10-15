import base64
import io
import json
import threading
import urllib.error
import urllib.request
from html.parser import HTMLParser

import pandas as pd
import pytest

import lambda_function
import local_api
from conftest import EXAMPLES, RAW_CSV
from mushroom_model import NUM_COLS, vocabulary


class FakeRuntime:
    def __init__(self, reply=None, fail=None):
        self.reply, self.fail, self.calls = reply, fail, []

    def invoke_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return {'Body': io.BytesIO(json.dumps(self.reply).encode())}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv('ENDPOINT_NAME', 'skooma-mushroom')
    def use(**kwargs):
        runtime = FakeRuntime(**kwargs)
        monkeypatch.setattr(lambda_function, 'runtime', runtime)
        return runtime
    return use


def call(event):
    result = lambda_function.handler(event, None)
    return result['statusCode'], json.loads(result['body'])


def test_lambda_forwards_the_body_to_the_endpoint(fake):
    runtime = fake(reply={'class': 'p'})
    assert call({'httpMethod': 'POST', 'body': '{"habitat": "d"}'}) == (200, {'class': 'p'})
    assert runtime.calls[0]['EndpointName'] == 'skooma-mushroom'
    assert runtime.calls[0]['ContentType'] == 'application/json'


def test_lambda_turns_model_errors_into_400(fake):
    fake(reply={'error': 'habitat must be one of'})
    assert call({'httpMethod': 'POST', 'body': '{"habitat": "zz"}'})[0] == 400


def test_lambda_hides_endpoint_failures_behind_502(fake):
    fake(fail=RuntimeError('secret internal detail'))
    status, body = call({'httpMethod': 'POST', 'body': '{}'})
    assert status == 502 and 'secret' not in body['error']


def test_lambda_handles_empty_and_base64_bodies(fake):
    fake(reply={'class': 'e'})
    assert call({'httpMethod': 'POST', 'body': None})[0] == 400
    assert call({'httpMethod': 'POST', 'body': '%%%not base64', 'isBase64Encoded': True})[0] == 400
    assert call({'httpMethod': 'POST', 'body': {'habitat': 'd'}})[0] == 400
    encoded = base64.b64encode(b'{"habitat": "d"}').decode()
    assert call({'httpMethod': 'POST', 'body': encoded, 'isBase64Encoded': True}) == (200, {'class': 'e'})


def test_lambda_separates_model_crashes_from_outages(fake):
    ModelError = type('ModelError', (Exception,), {})
    fake(fail=ModelError('boom'))
    assert call({'httpMethod': 'POST', 'body': '{}'})[0] == 500


def test_lambda_health_check_does_not_call_the_model(fake):
    runtime = fake(fail=RuntimeError('should not be called'))
    assert call({'httpMethod': 'GET'}) == (200, {'status': 'ok'})
    assert runtime.calls == []


def test_lambda_serves_the_page_without_calling_the_model(fake):
    runtime = fake(fail=RuntimeError('should not be called'))
    result = lambda_function.handler({'httpMethod': 'GET', 'resource': '/'}, None)
    assert result['statusCode'] == 200 and result['headers']['Content-Type'].startswith('text/html')
    assert '<form' in result['body'] and runtime.calls == []


def test_lambda_ask_route_predicts_like_predict(fake):
    fake(reply={'class': 'e'})
    assert call({'httpMethod': 'POST', 'resource': '/ask', 'body': '{"habitat": "d"}'}) == (200, {'class': 'e'})


class FormFields(HTMLParser):
    # collects the name of every input and, per select, the option values the page offers
    def __init__(self):
        super().__init__()
        self.inputs, self.selects, self.current = [], {}, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'input':
            self.inputs.append(attrs['name'])
        elif tag == 'select':
            self.current = attrs['name']
            self.selects[self.current] = []
        elif tag == 'option':
            self.selects[self.current].append(attrs['value'])


def test_page_offers_exactly_what_the_model_accepts():
    page = FormFields()
    page.feed(lambda_function.PAGE.read_text(encoding='utf-8'))
    vocab = vocabulary(pd.read_csv(RAW_CSV))
    assert page.inputs == list(NUM_COLS)
    assert set(page.selects) == set(vocab)
    for col, values in page.selects.items():
        # "don't know" first, then every letter the model knows, each once
        assert values[0] == '' and sorted(values[1:]) == vocab[col], col


@pytest.fixture
def server(trained):
    httpd = local_api.make_server(str(trained['dir']), '127.0.0.1', 0, 'test-key')
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()
    lambda_function.runtime = None


def http(url, body=None, key='test-key'):
    # same request the J backend will send with curl
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['x-api-key'] = key
    request = urllib.request.Request(url, data=body.encode() if body is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_local_api_end_to_end(server):
    assert http(server + '/health') == (200, {'status': 'ok'})
    status, answer = http(server + '/predict', (EXAMPLES / 'full.json').read_text())
    assert status == 200 and answer['class'] in ('e', 'p')
    status, answer = http(server + '/predict', (EXAMPLES / 'batch.json').read_text())
    assert status == 200 and len(answer) == 2
    status, answer = http(server + '/predict', (EXAMPLES / 'invalid.json').read_text())
    assert status == 400 and 'cap-diameter' in answer['error']


def test_local_api_checks_key_and_path(server):
    assert http(server + '/predict', '{}', key=None)[0] == 403
    assert http(server + '/predict', '{}', key='wrong')[0] == 403
    assert http(server + '/nothing', '{}') == (403, {'message': 'Missing Authentication Token'})
    assert http(server + '/predict')[0] == 403
    assert http(server + '/health', '{}')[0] == 403
    assert http(server + '/health', key=None)[0] == 403
    assert http(server + '/ask')[0] == 403
    assert http(server + '/', '{}')[0] == 403


def test_local_api_page_and_ask_work_without_a_key(server):
    with urllib.request.urlopen(server + '/', timeout=30) as r:
        assert r.status == 200 and r.headers['Content-Type'].startswith('text/html')
        assert 'Is this mushroom poisonous?' in r.read().decode()
    # the page sends every field, blanks as "" and numbers as text
    form = {'cap-diameter': '6.4', 'stem-height': '', 'stem-width': '', 'cap-shape': 'x', 'gill-color': '',
            'spore-print-color': '', 'stem-surface': 'f', 'ring-type': '', 'habitat': 'd', 'season': 'a'}
    status, answer = http(server + '/ask', json.dumps(form), key=None)
    assert status == 200 and answer['class'] in ('e', 'p') and 0 <= answer['probability_poisonous'] <= 1
    status, answer = http(server + '/ask', json.dumps({**form, 'cap-diameter': '-3'}), key=None)
    assert status == 400 and 'cap-diameter' in answer['error']
