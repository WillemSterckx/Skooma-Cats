import base64
import json
import os

runtime = None


def get_runtime():
    # created on first use, so tests and the local server can swap in their own runtime
    global runtime
    if runtime is None:
        import boto3
        from botocore.config import Config
        # give up before the 28 s lambda limit so the caller gets a clean 502 instead of a timeout
        runtime = boto3.client('sagemaker-runtime', config=Config(connect_timeout=3, read_timeout=24, retries={'total_max_attempts': 1}))
    return runtime


def respond(status, body):
    return {'statusCode': status, 'headers': {'Content-Type': 'application/json'}, 'body': json.dumps(body)}


def handler(event, context):
    method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method', 'POST')
    if method == 'GET':
        return respond(200, {'status': 'ok'})
    body = event.get('body') or ''
    try:
        if event.get('isBase64Encoded'):
            body = base64.b64decode(body, validate=True).decode('utf-8', errors='replace')
    except (ValueError, TypeError):
        return respond(400, {'error': 'request body is not valid base64'})
    if not isinstance(body, str) or not body.strip():
        return respond(400, {'error': 'empty request body, send the mushroom as JSON'})
    try:
        answer = get_runtime().invoke_endpoint(EndpointName=os.environ['ENDPOINT_NAME'], ContentType='application/json',
                                               Accept='application/json', Body=body.encode('utf-8'))
        result = json.loads(answer['Body'].read())
    except Exception as e:
        # the full error goes to the cloudwatch log, the caller only sees the type
        print('endpoint call failed:', repr(e))
        if type(e).__name__ == 'ModelError':
            return respond(500, {'error': 'the model crashed on this input, retrying will not help'})
        return respond(502, {'error': f'model endpoint failed ({type(e).__name__}), try again in a minute'})
    return respond(400 if isinstance(result, dict) and 'error' in result else 200, result)
