import numpy as np
import re
import time
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from src import config
from src.config import input_length, masterdata
from src.neural_network_trainer import load_model, \
    load_vocab_tokenizer, load_encoded_sentence_from_string, all_languages
from src.message import prepare_data, load_list_from_file

vocab_tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)
model = load_model(config.model_file_location, config.weights_file_location)


def to_language(binary_list):
    i = np.argmax(binary_list)
    
    return (all_languages[i], binary_list[0][i])


def get_neural_network_input(code):
    encoded_sentence = load_encoded_sentence_from_string(code, vocab_tokenizer)
    return pad_sequences([encoded_sentence], maxlen=input_length)


def detect(code):
    with tf.device('/device:GPU:0'):
        y_proba = model.predict(get_neural_network_input(code))
    return to_language(y_proba)


if __name__ == "__main__":
    masterdata['carriers']  =   list(map(lambda x: re.escape(x), load_list_from_file('carriers')))
    masterdata['stations']  =   list(map(lambda x: re.escape(x), load_list_from_file('stations')))
    masterdata['uldGroups'] =   list(map(lambda x: re.escape(x), load_list_from_file('uldGroups')))
    masterdata['message']   =   list(map(lambda x: re.escape(x), load_list_from_file('message')))


    code = """
<HEADER>
POSTEDDATE:=2018-07-30T11:58:19Z
FROM:=ZRHKUXH <ZRHKUXH@TYPEB.MCONNECT.AERO>
SUBJECT:=
</HEADER>
UCM
CX383/25JUL.BKQH.ZRH
IN
SI
ISOF PLA52398R7 / TK
SI UCM IN STD CHG MINUS 1 DAY
"""
    code = prepare_data(code)
    print(code)
    start = time.time()
    result = detect(code)
    end = time.time()

    print('Message Type: {}'.format(result[0]))
    print('Prob: {}%'.format(result[1] * 100))
    print('Time: {}'.format(str(end-start)))
