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

    with tf.Session() as sess:
        # Load saved model
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        # Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(model.predictions, feed_dict={model.inputs: [text]*trainer.batch_size, 
                                                    model.inputs_length: [len(text)]*trainer.batch_size,
                                                    model.targets_length: [len(text)+1], 
                                                    model.keep_prob: [1.0]})
        answer_texts = ["".join([int_to_vocab[i] for i in text if i not in pad]) for text in answer_logits]

    print('\nText')
    print('  Input Words: {} -> {}'.format(input_text.replace(' ', ''), ",".join(set([text.replace(' ', '') for text in answer_texts]))))
    print('  inputs:    [{}]'.format(",".join([str(i) for i in [text]*trainer.batch_size])))
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
    texts = ['AK8745']
    for text in texts:
        detect_spell(text)