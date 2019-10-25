from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, merge, Reshape
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional, Flatten, concatenate
from keras.layers.core import Masking

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

def get_tag_classify_model(model_name, word_context_length, char_feature_output, hidden_size, tag_classes_array):

    if model_name == 'BLSTM':
        char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
        merged = Masking(mask_value=0.,)(char_vector_input)
        x = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        main_loss = []
        for i in range(0, len(tag_classes_array)):
            main_loss1 = TimeDistributed(Dense(tag_classes_array[i], activation='softplus'))(x)
            main_loss1 = Activation('softmax')(main_loss1)
            main_loss.append(main_loss1)
        model = Model(input=[char_vector_input], output=main_loss)
        return model
