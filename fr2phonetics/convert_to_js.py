import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Concatenate
import tensorflowjs as tfjs

model = load_model('bidirectionnal_seq2seq_phonetics_fr.h5')

encoder_inputs = model.inputs[0]   # input_1
encoder_outputs, forward_h, forward_c, backward_h, backward_c = model.layers[1].output   # lstm_1
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.inputs[1]   # input_2
decoder_state_input_h = Input(shape=(64*2,), name='input_3')
decoder_state_input_c = Input(shape=(64*2,), name='input_4')
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

tfjs.converters.save_keras_model(encoder_model, 'encoder')
tfjs.converters.save_keras_model(decoder_model, 'decoder')
tfjs.converters.save_keras_model(model, 'bidirectionnal_seq2seq_phonetics_fr')