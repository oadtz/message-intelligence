import numpy as np
import pandas as pd
import tensorflow as tf
import os, shutil
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
import json
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# The default parameters
model_name = ''
epochs = 100
batch_size = int(os.getenv('BATCH_SIZE'))
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.9
keep_probability = 1.0
split_ratio = 0

def parse_args():
    parser = ArgumentParser(description='Train the spell correction engine')
    parser.add_argument('-m', '--model_name',
                        dest='model_name',
                        required=True)
    parser.add_argument('-e', '--epochs',
                        dest='epochs',
                        default='100')
    parser.add_argument('-p', '--prob',
                        dest='keep_probability',
                        default='1.0')
    parser.add_argument('-s', '--split_ratio',
                        dest='split_ratio',
                        default='0')
    parser.add_argument('-d', '--dir',
                        dest='directory')
    parser.add_argument('-f', '--file',
                        dest='file')
    parser.add_argument('-fc', '--file_correct_col',
                        dest='file_correct_col')
    parser.add_argument('-fe', '--file_error_col',
                        dest='file_error_col')
    parser.add_argument('-t', '--text',
                        dest='text')
    parser.add_argument('-te', '--text_error',
                        dest='text_error')
    parser.add_argument('-c', '--check_noisy_exists',
                        dest='check_noisy_exists',
                        default=False)
    args = parser.parse_args()
    
    return args.model_name, int(args.epochs), float(args.keep_probability), float(args.split_ratio), bool(args.check_noisy_exists), args.directory, args.file, args.file_correct_col, args.file_error_col, args.text, args.text_error

    
def noise_maker(word, threshold, check_noisy_exists):
    '''Relocate, remove, or add characters to create spelling mistakes'''
    
    noisy_word = []
    i = 0
    while i < len(word):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_word.append(word[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(word) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_word.append(word[i+1])
                    noisy_word.append(word[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_word.append(vocab_to_int[random_letter])
                noisy_word.append(word[i])
            # ~33% chance a character will not be typed
            else:
                pass     
        i += 1

    # Regenerate if word exists
    if check_noisy_exists and ints_to_vocab(noisy_word).strip().upper() != ints_to_vocab(word).strip().upper() and ints_to_vocab(noisy_word).strip().upper() in words:
        return noise_maker(word, threshold, check_noisy_exists)

    return noisy_word

def vocab_to_ints(word):
    int_word = []
    for character in word:
        int_word.append(vocab_to_int[character] if character in vocab_to_int.keys() else vocab_to_int['?'])
    
    return int_word

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


def pad_word_batch(word_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_word = max([len(word) for word in word_batch])
    return [word + [vocab_to_int['<PAD>']] * (max_word - len(word)) for word in word_batch]


def get_batches(words, batch_size, threshold):
    """Batch sentences, noisy sentences, and the lengths of their sentences together.
       With each epoch, sentences will receive new mistakes"""
    n = int(np.ceil(len(words)/batch_size))
    for batch_i in range(0, n):
        start_i = batch_i * batch_size
        words_batch = words[start_i:start_i + batch_size]
        
        words_batch_noisy = []
        for word in words_batch:
            word =  vocab_to_ints(error_words[ints_to_vocab(word)]) if ints_to_vocab(word) in error_words.keys() else word
            words_batch_noisy.append(noise_maker(word, threshold, check_noisy_exists))

        words_batch_eos = []
        for word in words_batch:
            word.append(vocab_to_int['<EOS>'])
            words_batch_eos.append(word)
            
        pad_words_batch = np.array(pad_word_batch(words_batch_eos))
        pad_words_noisy_batch = np.array(pad_word_batch(words_batch_noisy))
        
        # Need the lengths for the _lengths parameters
        pad_words_lengths = []
        for word in pad_words_batch:
            pad_words_lengths.append(len(word))
        
        pad_words_noisy_lengths = []
        for word in pad_words_noisy_batch:
            pad_words_noisy_lengths.append(len(word))


        yield pad_words_noisy_batch, pad_words_batch, pad_words_noisy_lengths, pad_words_lengths


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
    
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists("./resources/models/{}/saved_model.ckpt".format(model_name)):
            print('Checkpoint exists')
            saver.restore(sess, "./resources/models/{}/saved_model.ckpt".format(model_name))
        else:
            print('Checkpoint does not exist')
        
        graph = tf.get_default_graph()

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

                #print('Batch {}'.format(batch_i))
                #print('Input {}'.format(input_batch))
                #print('Target {}'.format(target_batch))
                #print('Start training for epoch_i = {} batch = {}'.format(epoch_i, batch_i))
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
                if testing_check > 0 and batch_i % testing_check == 0 and batch_i > 0:
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
                        checkpoint = "./resources/models/{}/saved_model.ckpt".format(model_name)
                        #saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break
            
            checkpoint = "./resources/models/{}/saved_model.ckpt".format(model_name)
            #saver = tf.train.Saver()
            saver.save(sess, checkpoint)

        freeze_graph(sess, graph)


def freeze_graph(sess, graph):
    #graph = tf.get_default_graph()
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


    dir_name = "./resources/models/{}/serve/".format(model_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    all_subdirs = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
    latest_ver = 0 if len(all_subdirs) == 0 else max(all_subdirs, key=os.path.basename)
    if not latest_ver:
        latest_ver = 0
    latest_ver = int(latest_ver) + 1

    builder = tf.saved_model.builder.SavedModelBuilder('./resources/models/{}/serve/{}/'.format(model_name, str(latest_ver)))
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })
    builder.save()

# In[ ]:

if __name__ == "__main__":
    # Get vars from arguments
    model_name, epochs, keep_probability, split_ratio, check_noisy_exists, directory, file, file_correct_col, file_error_col, text, text_error = parse_args()

    # Get words
    words = []
    error_words = {}
    if text:
        print('Training for {}'.format(text))
        #batch_size = 1
        words = [t.strip().upper() for t in text.split(',')]
        if text_error:
            error_words = dict(zip([t.strip().upper() for t in text.split(',')], [t.strip().upper() for t in text_error.split(',')]))
    elif file:
        print('Training for file {}'.format(file))
        if file_correct_col and file_error_col:
            f = pd.read_csv(file)
            words = [text.strip().upper() for text in f[file_correct_col].tolist()]
            error_words = { row[file_correct_col]:row[file_error_col] for i, row in f.iterrows() }
        else:
            with open(file) as f:
                words = [text.strip().upper() for text in f.read().splitlines()]
                f.close()
    elif directory:
        print('Training for dir {}'.format(directory))
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for file in files:
            with open(file, 'r') as f:
                words.append(f.read())
                f.close()
    else:
        print ('Missing input source')
        exit()

    print("There are {} words.".format(len(words)))

    # Create a dictionary to convert the vocabulary (characters) to integers
    vocab_to_int = {
        '\n': 0,
        '\r': 1
    }
    count = 2
    for i in range(32, 127):
        vocab_to_int[chr(i)] = count
        count += 1

    # Add special tokens to vocab_to_int
    codes = ['<PAD>','<EOS>','<GO>']
    for code in codes:
        vocab_to_int[code] = count
        count += 1
    
    model_dir = './resources/models/{}'.format(model_name)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'vocabs.json'), 'w') as f:
        json.dump(vocab_to_int, f)
        f.close()

    # Check the size of vocabulary and all of the values
    vocab_size = len(vocab_to_int)
    print("The vocabulary contains {} characters.".format(vocab_size))
    print(sorted(vocab_to_int))

    # Convert ULDs to integers
    int_words = []

    for word in words:
        int_words.append(vocab_to_ints(word))

    # Split the data into training and testing sentences
    training, testing = train_test_split(int_words, test_size = split_ratio, random_state = 2)
    #training = training + training[0 : batch_size - (len(training)%batch_size)]
    #training = training + ([[vocab_to_int['<EOS>']] * len(max(training, key=len))] * (batch_size - (len(training)%batch_size)))
    n_batch = int(np.ceil(len(training) / batch_size))
    training = training + training * (batch_size//len(training) + 1) 
    training = training[0: (n_batch * batch_size)]

    # Sort the flihgt no by length to reduce padding, which will allow the model to train faster
    training_sorted = []
    testing_sorted = []

    for i in range(1, len(max(words, key=len)) + 1):
        for word in training:
            if len(word) == i:
                training_sorted.append(word)
        for word in testing:
            if len(word) == i:
                testing_sorted.append(word)

    print("Number of training words:", len(training_sorted))
    #print("Number of testing words:", len(testing_sorted))

    # Check to ensure the sentences have been selected and sorted correctly
    #for i in range(5):
    #    print(training_sorted[i], len(training_sorted[i]))


    letters = list(vocab_to_int.keys())[:len(vocab_to_int.keys()) - 3]

    # Train the model with the desired tuning parameters
    with tf.device('/device:GPU:0'):
        for k in [keep_probability]:
            for num_layers in [2]:
                for threshold in [0.9]:
                    log_string = 'kp={},nl={},th={}'.format(k,
                                                            num_layers,
                                                            threshold) 
                    model = build_graph(k, rnn_size, num_layers, batch_size, 
                                        learning_rate, embedding_size, direction, vocab_to_int)
                    train(model, epochs, log_string)
