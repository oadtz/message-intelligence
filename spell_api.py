import json
import numpy as np
# Communication to TensorFlow server via gRPC
import grpc
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2
from flask import Flask
from flask import request
from flask import jsonify

HOST = 'localhost:9000'

app = Flask(__name__)

def convert_data(raw_data):
    return np.array(raw_data, dtype=np.float32)

def output_format(output):

    x = output.tensor_shape.dim[0].size
    y = output.tensor_shape.dim[1].size

    output = np.unique(np.reshape(output.int_val, (x, y)), axis=0)

    return output

def make_tensor_proto_int(data, shape):
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in shape]
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_INT32,
        tensor_shape=tensor_shape_proto,
        int_val=np.array(data).ravel().tolist())
    
    return tensor_proto

def make_tensor_proto_float(data, shape):
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in shape]
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_FLOAT,
        tensor_shape=tensor_shape_proto,
        float_val=np.array(data).ravel().tolist())
    
    return tensor_proto

@app.route("/flights", methods=['GET'])
def get_prediction():
    req_data = request.args
    text = req_data['text']
    prob = req_data['prob']

    texts = text.split(',')
    prob = float(prob) if prob else 1

    prediction = flights_predict(texts, prob)

    return jsonify(prediction)

def flights_predict(texts, prob):
    return predict('flights', texts, prob)
    
def predict(model, texts, prob):
    channel = grpc.insecure_channel(HOST)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    vocab_to_int = {}
    with open('./resources/models/{}/vocabs.json'.format(model)) as f:
        vocab_to_int = json.load(f)
        f.close()

    # Create another dictionary to convert integers to their respective characters
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character

    pad = [vocab_to_int["<PAD>"], vocab_to_int["<EOS>"]] 

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = 'serving_default'

    answers = []
    for text in texts:
        text = [vocab_to_int[word] for word in text]
        batch_size = 128

        inputs = [text]*batch_size
        inputs_length = [len(text)]*batch_size
        targets_length =[len(text)+1]
        
        request.inputs['inputs'].CopyFrom(make_tensor_proto_int(inputs, shape=[batch_size, len(text)]))
        request.inputs['inputs_length'].CopyFrom(make_tensor_proto_int(inputs_length, shape=[batch_size]))
        request.inputs['targets_length'].CopyFrom(make_tensor_proto_int(targets_length, shape=[len(text) + 1]))
        request.inputs['keep_prob'].CopyFrom(make_tensor_proto_float(prob, shape=[1]))


        result = stub.Predict(request, 60.0)  # 60 secs timeout

        result = output_format(result.outputs['outputs'])

        for text in result:
            answers.append("".join([int_to_vocab[i] for i in text if i not in pad]))
    
    return answers
        
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9001)