import numpy as np
from predict_client.prod_client import ProdClient
from flask import Flask
from flask import request
from flask import jsonify

HOST = 'localhost:9000'
MODEL_NAME = 'flight_spell'
MODEL_VERSION = 1

app = Flask(__name__)
client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION)


def get_prediction_from_model(data):
    req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': data}]

    prediction = client.predict(req_data, request_timeout=10)

    return prediction


@app.route("/flight-spell", methods=['get'])
def get_prediction():
    req_data = request.get_json()
    raw_data = req_data['flight']

    prediction = get_prediction_from_model(raw_data)

    # ndarray cannot be converted to JSON
    return jsonify({ 'predictions': prediction['outputs'].tolist() })

if __name__ == '__main__':
    app.run(host='localhost',port=3000)