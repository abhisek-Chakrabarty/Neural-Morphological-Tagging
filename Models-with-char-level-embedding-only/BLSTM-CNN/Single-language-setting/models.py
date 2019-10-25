from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, merge, Reshape
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional, Flatten, concatenate
from keras.layers.core import Masking
#from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPooling1D

def get_char_model(model_name, max_word_length, embedded_char_vector_length, char_feature_output):
    if model_name == 'BLSTM':
        char_input = Input(shape=(max_word_length,), dtype='float32', name='char_input')
        char_input1 = Embedding(1000, embedded_char_vector_length, input_length=max_word_length)(char_input)
        char_input2 = Dropout(0.2)(char_input1)
        lstm_out_forward = LSTM(char_feature_output, dropout_W=0.2, dropout_U=0.2)(char_input2)
        lstm_out_backward = LSTM(char_feature_output, dropout_W=0.2, dropout_U=0.2, go_backwards=True)(char_input2)
        merged = concatenate([lstm_out_forward, lstm_out_backward], axis=1)
        model = Model(input=[char_input], output=[merged])
        return model




def get_tag_classify_model_with_CNN(word_context_length, char_feature_output, no_of_kernels, filter_window_length, tag_classes):
    char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
    reshape = Reshape((word_context_length * char_feature_output, 1))(char_vector_input)
    conv_layer = Conv1D(no_of_kernels, ((2*filter_window_length+1) * char_feature_output,), strides = (char_feature_output), activation='relu', name='conv_output')(reshape)
    main_loss1 = TimeDistributed(Dense(tag_classes, activation='softplus'))(conv_layer)
    main_loss = Activation('softmax')(main_loss1)
    model = Model(input=[char_vector_input], output=[main_loss])
    intermediate_layer_model = Model(input=[char_vector_input], output=[model.get_layer('conv_output').output])
    return model, intermediate_layer_model

