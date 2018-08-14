import json
import tensorflow as tf
from src import flights_trainer as trainer


def text_to_ints(text):
    '''Prepare the text for the model'''
    
    text = [vocab_to_int[word] for word in text]
    text.append(vocab_to_int['<EOS>'])

    return text

def detect_spell(text):
    # Remove the padding from the generated sentence
    pad = [vocab_to_int["<PAD>"], vocab_to_int["<EOS>"]] 
    text = text_to_ints(text.upper())
    input_text = "".join([int_to_vocab[i] for i in text if i not in pad])

    checkpoint = "./resources/models/flight_spell.ckpt"


    model = trainer.build_graph(trainer.keep_probability, trainer.rnn_size, trainer.num_layers, trainer.batch_size, trainer.learning_rate, trainer.embedding_size, trainer.direction, vocab_to_int) 

    frozen_graph = './resources/models/flight_spell.pb'

    with tf.gfile.GFile(frozen_graph, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            restored_graph_def,
            name=''
        )
    
    with tf.Session(graph=graph) as sess:
    #with tf.Session() as sess:
        # Load saved model
        #saver = tf.train.Saver()
        #saver.restore(sess, checkpoint)
        inputs = graph.get_tensor_by_name("inputs/inputs:0")
        inputs_length = graph.get_tensor_by_name("inputs_length:0")
        targets_length = graph.get_tensor_by_name("targets_length:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        predictions = graph.get_tensor_by_name("predictions/predictions:0")

        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(predictions, feed_dict={inputs: [text]*trainer.batch_size, 
                                                    inputs_length: [len(text)]*trainer.batch_size,
                                                    targets_length: [len(text)+1], 
                                                    keep_prob: [0.90]})
        answer_texts = ["".join([int_to_vocab[i] for i in text if i not in pad]) for text in answer_logits]

    print('\nText')
    print('  Input Words: {} -> {}'.format(input_text.replace(' ', ''), ",".join(set([text.replace(' ', '') for text in answer_texts]))))
    print('  inputs:    [{}]'.format(",".join([str(i) for i in text])))
    print('  inputs_length:    [{}]'.format(",".join([str(i) for i in [len(text)]*trainer.batch_size])))
    print('  targets_length:    [{}]'.format(",".join([str(i) for i in [len(text)+1]])))


if __name__ == "__main__":
    vocab_to_int = []
    with open('./resources/flights_to_int.json') as f:
        vocab_to_int = json.load(f)
        f.close()


    # Create another dictionary to convert integers to their respective characters
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character
    # Load all flights
    flights = []
    with open('./masterdata/flights', 'r') as f:
        flights = f.read().splitlines()
    # Create your own sentence or use one from the dataset
    texts = ['C8 0234', 'XT 853AD', 'SV 0547', 'SK 0547', 'AC 6271T']
    for text in texts:
        detect_spell(text)