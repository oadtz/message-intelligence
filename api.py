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
from grpc.beta import implementations
import tensorflow as tf
from src import flights_trainer as trainer

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto

from tensorflow.contrib.util import make_tensor_proto

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

    host, port = args.server.split(':')
    
    return host, port, args.text

def text_to_ints(text):
    '''Prepare the text for the model'''
    
    text = [vocab_to_int[word] for word in text]
    text.append(vocab_to_int['<EOS>'])

    return text


def main():
    # parse command line arguments
    host, port, text = parse_args()

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

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

    trainer.batch_size = 1

    inputs = [text]*trainer.batch_size
    inputs_length = [len(text)]*trainer.batch_size
    targets_length =[len(text)+1]
    
    request.inputs['inputs'].CopyFrom(make_tensor_proto(inputs, shape=[len(text) * trainer.batch_size], dtype=tf.int32))
    request.inputs['inputs_length'].CopyFrom(make_tensor_proto(inputs_length, shape=[trainer.batch_size]))
    request.inputs['targets_length'].CopyFrom(make_tensor_proto(targets_length, shape=[len(text) + 1]))
    request.inputs['keep_prob'].CopyFrom(make_tensor_proto(0.95, shape=[1]))

    result = stub.Predict(request, 60.0)  # 60 secs timeout
    print(result)

    end = time.time()
    time_diff = end - start
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    vocab_to_int = []
    with open('./resources/flights_to_int.json') as f:
        vocab_to_int = json.load(f)
        f.close()

    main()