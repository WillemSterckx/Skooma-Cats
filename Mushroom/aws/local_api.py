import argparse
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'model'))

import inference  # noqa: E402
import lambda_function  # noqa: E402


class LocalRuntime:
    # behaves like boto3's sagemaker-runtime client, but runs inference.py in this process
    def __init__(self, model_dir):
        self.model = inference.model_fn(model_dir)

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        prediction = inference.predict_fn(inference.input_fn(Body, ContentType), self.model)
        body, _ = inference.output_fn(prediction, Accept)
        return {'Body': io.BytesIO(body.encode('utf-8'))}


class Handler(BaseHTTPRequestHandler):
    api_key = None

    def answer(self, method):
        # same routes and answers as the api gateway stage in gateway.yaml, which says 403 for a route it doesn't have
        if (method, self.path.split('?')[0]) not in (('POST', '/predict'), ('GET', '/health')):
            return self.send(403, {'message': 'Missing Authentication Token'})
        if self.headers.get('x-api-key') != self.api_key:
            return self.send(403, {'message': 'Forbidden'})
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            return self.send(400, {'message': 'bad Content-Length'})
        body = self.rfile.read(length).decode('utf-8', errors='replace')
        result = lambda_function.handler({'httpMethod': method, 'body': body, 'isBase64Encoded': False}, None)
        self.send(result['statusCode'], json.loads(result['body']))

    def send(self, status, payload):
        data = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.answer('GET')

    def do_POST(self):
        self.answer('POST')


def make_server(model_dir, host, port, api_key):
    lambda_function.runtime = LocalRuntime(model_dir)
    os.environ.setdefault('ENDPOINT_NAME', 'local')
    Handler.api_key = api_key
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=str(HERE.parent / 'artifacts' / 'model'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--api-key', default='local-dev-key')
    args = parser.parse_args()
    server = make_server(args.model_dir, args.host, args.port, args.api_key)
    print(f'mushroom model api on http://{args.host}:{args.port}/predict (header x-api-key: {args.api_key})')
    server.serve_forever()
