
# coding: utf-8

# # Creating a Spell Checker

# The objective of this project is to build a model that can take a sentence with spelling mistakes as input, and output the same sentence, but with the mistakes corrected. The data that we will use for this project will be twenty popular books from [Project Gutenberg](http://www.gutenberg.org/ebooks/search/?sort_order=downloads). Our model is designed using grid search to find the optimal architecture, and hyperparameter values. The best results, as measured by sequence loss with 15% of our data, were created using a two-layered network with a bi-direction RNN in the encoding layer and Bahdanau Attention in the decoding layer. [FloydHub's](https://www.floydhub.com/) GPU service was used to train the model.
# 
# The sections of the project are:
# - Loading the Data
# - Preparing the Data
# - Building the Model
# - Training the Model
# - Fixing Custom Sentences
# - Summary

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import os, shutil
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
import json
from sklearn.model_selection import train_test_split

# The default parameters
data_path = '/masterdata/' if os.path.isdir('/masterdata/') else './masterdata'
epochs = 100
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.9
keep_probability = 1.0


def noise_maker(flight, threshold):
    '''Relocate, remove, or add characters to create spelling mistakes'''
    
    noisy_flight = []
    i = 0
    while i < len(flight):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_flight.append(flight[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(flight) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_flight.append(flight[i+1])
                    noisy_flight.append(flight[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_flight.append(vocab_to_int[random_letter])
                noisy_flight.append(flight[i])
            # ~33% chance a character will not be typed
            else:
                pass     
        i += 1

    return noisy_flight

def ints_to_vocab(ints):
    
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character
    
    return "".join([int_to_vocab[i] for i in ints if int_to_vocab[i] not in codes])

# # Building the Model
def model_inputs():
    '''Create palceholders for inputs to the model'''
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


def process_encoding_input(targets, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input



def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    '''Create the encoding layer'''
    
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop, 
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state
        
        
    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                            input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                            input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                            cell_bw, 
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output,2)
            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0]

def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_target_length):
    '''Create the training logits'''
    
    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

        training_logits,_ ,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)
        return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_target_length, batch_size):
    '''Create the inference logits'''
    
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits ,_ ,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)

        return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, 
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  inputs_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')
    
    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                        attn_mech,
                                                        rnn_size)
    initial_state = dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=enc_state)

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, 
                                                  targets_length, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  
                                                    vocab_to_int['<GO>'], 
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size)

    return training_logits, inference_logits


def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):
    '''Use the previous functions to create the training and inference logits'''
    
    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers, 
                                           enc_embed_input, keep_prob, direction)
    
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        dec_embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        inputs_length, 
                                                        targets_length, 
                                                        max_target_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers,
                                                        direction)
    
    return training_logits, inference_logits


def pad_flight_batch(flight_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_flight = max([len(flight) for flight in flight_batch])
    return [flight + [vocab_to_int['<PAD>']] * (max_flight - len(flight)) for flight in flight_batch]


def get_batches(flights, batch_size, threshold):
    """Batch sentences, noisy sentences, and the lengths of their sentences together.
       With each epoch, sentences will receive new mistakes"""
    
    for batch_i in range(0, len(flights)//batch_size):
        start_i = batch_i * batch_size
        flights_batch = flights[start_i:start_i + batch_size]
        
        flights_batch_noisy = []
        for flight in flights_batch:
            flights_batch_noisy.append(noise_maker(flight, threshold))

        flights_batch_eos = []
        for flight in flights_batch:
            flight.append(vocab_to_int['<EOS>'])
            flights_batch_eos.append(flight)
            
        pad_flights_batch = np.array(pad_flight_batch(flights_batch_eos))
        pad_flights_noisy_batch = np.array(pad_flight_batch(flights_batch_noisy))
        
        # Need the lengths for the _lengths parameters
        pad_flights_lengths = []
        for flight in pad_flights_batch:
            pad_flights_lengths.append(len(flight))
        
        pad_flights_noisy_lengths = []
        for flight in pad_flights_noisy_batch:
            pad_flights_noisy_lengths.append(len(flight))


        yield pad_flights_noisy_batch, pad_flights_batch, pad_flights_noisy_lengths, pad_flights_lengths


# *Note: This set of values achieved the best results.*


def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction, vocab_to_int):
    tf.reset_default_graph()
    
    # Load the model inputs    
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, 
                                                targets, 
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


# ## Training the Model
def train(model, epochs, log_string):
    '''Train the RNN'''


    with tf.Session() as sess:
    
        if tf.train.checkpoint_exists("./resources/models/flight_spell/saved_model.ckpt"):
            print('Checkpoint exists')
            #saver = tf.train.import_meta_graph('./resources/models/flight_spell.ckpt.meta', clear_devices=True)
            saver = tf.train.Saver()
            saver.restore(sess, "./resources/models/flight_spell/saved_model.ckpt")
        else:
            print('Checkpoint does not exist')
            sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        testing_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0
        
        display_step = 30 # The progress of the training will be displayed after every 30 batches
        stop_early = 0 
        stop = 3 # If the batch_loss_testing does not decrease in 3 consecutive checks, stop training
        per_epoch = 3 # Test the model 3 times per epoch
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))

        for epoch_i in range(1, epochs+1): 
            batch_loss = 0
            batch_time = 0
            
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()

                summary, loss, _ = sess.run([model.merged,
                                             model.cost, 
                                             model.train_op], 
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: keep_probability})


                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time

                # Record the progress of training
                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(training_sorted) // batch_size, 
                                  batch_loss / display_step, 
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                #### Testing ####
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(testing_sorted, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost], 
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})

                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        # Record the progress of testing
                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch_i + 1
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing, 
                                  batch_time_testing))
                    
                    batch_time_testing = 0

                    # If the batch_loss_testing is at a new minimum, save the model
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!') 
                        stop_early = 0
                        checkpoint = "./resources/models/flight_spell/saved_model.ckpt"
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break

        freeze_graph(sess)


def freeze_graph(sess):
    graph = tf.get_default_graph()
    # Freezing graph to use with Tensorflow Serving
    inputs = graph.get_tensor_by_name("inputs/inputs:0")
    inputs_length = graph.get_tensor_by_name("inputs_length:0")
    targets_length = graph.get_tensor_by_name("targets_length:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    predictions = graph.get_tensor_by_name("predictions/predictions:0")

    model_input = {
        'inputs': tf.saved_model.utils.build_tensor_info(inputs),
        'inputs_length': tf.saved_model.utils.build_tensor_info(inputs_length),
        'targets_length': tf.saved_model.utils.build_tensor_info(targets_length),
        'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)
    }
    model_output = tf.saved_model.utils.build_tensor_info(predictions)

    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=model_input,
        outputs={'outputs': model_output},
        method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)


    dir_name = "./resources/models/flight_spell/serve/"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    builder = tf.saved_model.builder.SavedModelBuilder('./resources/models/flight_spell/serve/1/')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })
    builder.save()

# In[ ]:

if __name__ == "__main__":
    flights = []
    with open(os.path.join(data_path, 'flights')) as f:
        flights = f.read().splitlines()

    for i in range(len(flights)):
        flights[i] = str(flights[i]).strip().upper()

    print("There are {} Flights.".format(len(flights)))

    # Create a dictionary to convert the vocabulary (characters) to integers
    vocab_to_int = {}
    count = 0
    '''
    for flight in flights:
        for character in flight:
            if character not in vocab_to_int:
                vocab_to_int[character] = count
                count += 1
    '''
    for i in range(ord('A'), ord('Z') + 1):
        vocab_to_int[chr(i)] = count
        count += 1
    for i in range(ord('0'), ord('9') + 1):
        vocab_to_int[chr(i)] = count
        count += 1

    # Add special tokens to vocab_to_int
    codes = ['<PAD>','<EOS>','<GO>']
    for code in codes:
        vocab_to_int[code] = count
        count += 1
    with open('./resources/flights_to_int.json', 'w') as f:
        json.dump(vocab_to_int, f)
        f.close()

    # Check the size of vocabulary and all of the values
    vocab_size = len(vocab_to_int)
    print("The vocabulary contains {} characters.".format(vocab_size))
    print(sorted(vocab_to_int))

    # Convert ULDs to integers
    int_flights = []

    for flight in flights:
        int_flight = []
        for character in flight:
            int_flight.append(vocab_to_int[character])
        int_flights.append(int_flight)

    # Split the data into training and testing sentences
    training, testing = train_test_split(int_flights, test_size = 0, random_state = 2)
    # training = int_flights # Set training set to be 100%


    # Sort the flihgt no by length to reduce padding, which will allow the model to train faster
    training_sorted = []
    testing_sorted = []

    for i in range(4, 12):
        for flight in training:
            if len(flight) == i:
                training_sorted.append(flight)
        for flight in testing:
            if len(flight) == i:
                testing_sorted.append(flight)

    print("Number of training Flights:", len(training_sorted))
    print("Number of testing Flights:", len(testing_sorted))


    # In[24]:

    # Check to ensure the sentences have been selected and sorted correctly
    for i in range(5):
        print(training_sorted[i], len(training_sorted[i]))


    letters = list(vocab_to_int.keys())[:len(vocab_to_int.keys()) - 3]

    # Check to ensure noise_maker is making mistakes correctly.
    '''
    threshold = 0.9
    for flight in training[:5]:
        print(flight)
        print(noise_maker(flight, threshold))
        print()
    '''

    # Train the model with the desired tuning parameters
    with tf.device('/device:GPU:0'):
        for keep_probability in [1.0]:
            for num_layers in [2]:
                for threshold in [0.9]:
                    log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                            num_layers,
                                                            threshold) 
                    model = build_graph(keep_probability, rnn_size, num_layers, batch_size, 
                                        learning_rate, embedding_size, direction, vocab_to_int)
                    train(model, epochs, log_string)

