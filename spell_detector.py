import json
import numpy as np

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
import grpc
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2


def parse_args():
    parser = ArgumentParser(description='Request a TensorFlow server for a spell prediction on the text')
    parser.add_argument('-m', '--model_name',
                        dest='model_name',
                        required=True)
    parser.add_argument('-s', '--server',
                        dest='server',
                        default='localhost:9000',
                        help='Prediction service host:port')
    parser.add_argument('-t', '--text',
                        dest='text',
                        default='',
                        help='Text to be checked')
    parser.add_argument('-p', '--prob',
                        dest='prob',
                        default='1',
                        help='Probability')
    parser.add_argument('-v', '--debug',
                        dest='debug',
                        default=False,
                        help='Print debug')
    args = parser.parse_args()
    
    return args.model_name, args.server, args.text, float(args.prob), bool(args.debug)

def text_to_ints(text):
    '''Prepare the text for the model'''
    
    text = [vocab_to_int[word] for word in text]
    text.append(vocab_to_int['<EOS>'])

    return text

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

def predict():
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Create another dictionary to convert integers to their respective characters
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character

    pad = [vocab_to_int["<PAD>"], vocab_to_int["<EOS>"]]

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'

    for text in texts:
        text = text_to_ints(text.upper())
        batch_size = 1

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
            print("".join([int_to_vocab[i] for i in text if i not in pad]))
        

        if debug:
            payload = json.dumps({
                "instances": [
                    {
                        "inputs": np.array(inputs).ravel().tolist(),
                        "inputs_length": np.array(inputs_length).ravel().tolist(),
                        "targets_length": np.array(targets_length).ravel().tolist(),
                        "keep_prob": np.array(prob).ravel().tolist()
                    }
                ]
            })
            print(payload)


if __name__ == '__main__':
    # parse command line arguments
    model_name, server, text, prob, debug = parse_args()

    texts = text.split(',')

    vocab_to_int = []
    with open('./resources/models/{}/vocabs.json'.format(model_name)) as f:
        vocab_to_int = json.load(f)
        f.close()

    predict()