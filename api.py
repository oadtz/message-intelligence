'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''
import json
import time
import numpy as np

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
import grpc
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

from os import listdir
from os.path import isfile, join


def parse_args():
    parser = ArgumentParser(description='Request a TensorFlow server for a prediction on the flight nbr')
    parser.add_argument('-s', '--server',
                        dest='server',
                        default='localhost:9000',
                        help='prediction service host:port')
    parser.add_argument('-t', '--text',
                        dest='text',
                        default='',
                        help='flight nbr')
    args = parser.parse_args()

    #host, port = args.server.split(':')
    
    print('Text = {}'.format(args.text))
    return args.server, args.text

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

def main():
    # parse command line arguments
    server, text = parse_args()

    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    start = time.time()

    pad = [vocab_to_int["<PAD>"], vocab_to_int["<EOS>"]] 
    text = text_to_ints(text.upper())

    # Create another dictionary to convert integers to their respective characters
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'flight_spell'
    request.model_spec.signature_name = 'serving_default'

    inputs = [text]*128
    inputs_length = [len(text)]*128
    targets_length =[len(text)+1]
    
    request.inputs['inputs'].CopyFrom(make_tensor_proto_int(inputs, shape=[128, len(text)]))
    #print('inputs:    [{}]'.format(",".join([str(i) for i in inputs])))
    request.inputs['inputs_length'].CopyFrom(make_tensor_proto_int(inputs_length, shape=[128]))
    #print('inputs_length:    [{}]'.format(",".join([str(i) for i in inputs_length)))
    request.inputs['targets_length'].CopyFrom(make_tensor_proto_int(targets_length, shape=[len(text) + 1]))
    #print('targets_length:    [{}]'.format(",".join([str(i) for i in targets_length])))
    request.inputs['keep_prob'].CopyFrom(make_tensor_proto_float(0.95, shape=[1]))

    result = stub.Predict(request, 60.0)  # 60 secs timeout

    result = output_format(result.outputs['outputs'])

    print ('Output = ')
    for text in result:
        print('\t' + "".join([int_to_vocab[i] for i in text if i not in pad]))

    end = time.time()
    time_diff = end - start
    #print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    vocab_to_int = []
    with open('./resources/flights_to_int.json') as f:
        vocab_to_int = json.load(f)
        f.close()

    main()