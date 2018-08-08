import json
import tensorflow as tf
from src import flights_trainer as trainer


def text_to_ints(text):
    '''Prepare the text for the model'''
    
    return [vocab_to_int[word] for word in text]


# In[176]:

def detect_spell(text):
    text = text_to_ints(text.upper())

    #random = np.random.randint(0,len(testing_sorted))
    #text = testing_sorted[random]
    #text = noise_maker(text, 0.95)

    checkpoint = "./resources/models/flight_spell.ckpt"

    model = trainer.build_graph(trainer.keep_probability, trainer.rnn_size, trainer.num_layers, trainer.batch_size, trainer.learning_rate, trainer.embedding_size, trainer.direction, vocab_to_int) 

    with tf.Session() as sess:
        # Load saved model
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        
        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(model.predictions, {model.inputs: [text]*trainer.batch_size, 
                                                    model.inputs_length: [len(text)]*trainer.batch_size,
                                                    model.targets_length: [len(text)+1], 
                                                    model.keep_prob: [1.0]})[0]

    # Remove the padding from the generated sentence
    pad = vocab_to_int["<PAD>"] 

    print('\nText')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

    print('\nSummary')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


if __name__ == "__main__":
    vocab_to_int = []
    with open('./resources/flights_to_int.json') as f:
        vocab_to_int = json.load(f)
        f.close()


    # Create another dictionary to convert integers to their respective characters
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character
    # Create your own sentence or use one from the dataset
    texts = ['C80234', 'XT853AD', 'S0547', 'SV0547']
    for text in texts:
        detect_spell(text)