from keras.models import Model, load_model
from keras.layers import  Input, Concatenate
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image


input_token_path = './fr_chars.txt'
target_token_path = './fr_phonetic_chars.txt'

with open(input_token_path,'r', encoding='UTF8') as file:
  input_token_index = json.loads(file.read())

with open(target_token_path,'r', encoding='UTF8') as file:
  target_token_index = json.loads(file.read())

reverse_input_char_index = {i: char for i, char in enumerate(input_token_index)}
reverse_target_char_index = {i: char for i, char in enumerate(target_token_index)}

max_encoder_seq_length = 75 #don't change
max_decoder_seq_length = 75 #don't change
num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)
latent_dim = 64

def encodeText(x):
    encoder_input_data = np.zeros((len(x), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for i, input_text in enumerate(x):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.  
    return encoder_input_data

def showWordVector(word):
    data, _, _ = encodeText(word)
    plt.imshow(data*255, cmap='gray', vmin=0, vmax=255)
    plt.show()

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

model = load_model("./bidirectionnal_seq2seq_phonetics_fr.h5")

encoder_inputs = model.inputs[0]   # input_1
encoder_outputs, forward_h, forward_c, backward_h, backward_c = model.layers[1].output   # lstm_1
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.inputs[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim*2,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim*2,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[5]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[-1]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

x_temp = ["bleu","couleur", "lumière", "oeil", "vue",
          "ouïe", "oreille interne", "système nerveux", "métabolisme", "humain",
          "système solaire", "galaxie", "protons et neutrons", "particule élémentaire", "mon cher watson"]

x_temp = [x_temp[i].lower() for i in range(len(x_temp))]
encoder_temp_input_data = encodeText(x_temp)

for seq_index in range(len(encoder_temp_input_data)):
    input_seq = encoder_temp_input_data[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print()
    print('Data #{} / {}'.format(seq_index+1,len(encoder_temp_input_data)))
    print("input sentence : {}".format(x_temp[seq_index]))
    print("decoded sentence : {}".format(decoded_sentence[:-1]))