from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, merge, Reshape
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional, Flatten, concatenate
from keras.layers.core import Masking
from keras.layers.convolutional import Convolution1D, MaxPooling1D

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

    elif model_name == 'BGRU':
        char_input = Input(shape=(max_word_length,), dtype='float32', name='char_input')
        char_input1 = Embedding(1000, embedded_char_vector_length, input_length=max_word_length)(char_input)
        char_input2 = Dropout(0.2)(char_input1)
        gru_out_forward = GRU(char_feature_output, dropout_W=0.2, dropout_U=0.2)(char_input2)
        gru_out_backward = GRU(char_feature_output, dropout_W=0.2, dropout_U=0.2, go_backwards=True)(char_input2)
        merged = merge([gru_out_forward, gru_out_backward], mode='concat', concat_axis=1)
        model = Model(input=[char_input], output=[merged])
        return model

    elif model_name == 'SimpleRNN':
        char_input = Input(shape=(max_word_length,), dtype='float32', name='char_input')
        char_input1 = Embedding(1000, embedded_char_vector_length, input_length=max_word_length)(char_input)
        char_input2 = Dropout(0.2)(char_input1)
        SimpleRNN_out_forward = SimpleRNN(char_feature_output, dropout_W=0.2, dropout_U=0.2)(char_input2)
        SimpleRNN_out_backward = SimpleRNN(char_feature_output, dropout_W=0.2, dropout_U=0.2, go_backwards=True)(char_input2)
        merged = merge([SimpleRNN_out_forward, SimpleRNN_out_backward], mode='concat', concat_axis=1)
        model = Model(input=[char_input], output=[merged])
        return model





def get_tag_classify_model_with_CNN(word_context_length, char_feature_output, no_of_kernels, filter_window_length, tag_classes1, tag_classes2, tag_classes3):
    char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
    reshape = Reshape((word_context_length * char_feature_output, 1))(char_vector_input)
    conv_layer = Convolution1D(no_of_kernels, ((2*filter_window_length+1) * char_feature_output,), strides = (char_feature_output), activation='relu')(reshape)
    main_loss1 = TimeDistributed(Dense(tag_classes1, activation='softplus'))(conv_layer)
    main_loss2 = TimeDistributed(Dense(tag_classes2, activation='softplus'))(conv_layer)
    main_loss3 = TimeDistributed(Dense(tag_classes3, activation='softplus'))(conv_layer)
    main_loss1 = Activation('softmax')(main_loss1)
    main_loss2 = Activation('softmax')(main_loss2)
    main_loss3 = Activation('softmax')(main_loss3)
    model = Model(input=[char_vector_input], output=[main_loss1, main_loss2, main_loss3])
    return model




def get_tag_classify_model(model_name, word_context_length, char_feature_output, hidden_size, tag_classes):

    if model_name == 'BLSTM':
        char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
        merged = Masking(mask_value=0.,)(char_vector_input)
        x = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        main_loss1 = TimeDistributed(Dense(tag_classes, activation='softplus'))(x)
        main_loss = Activation('softmax')(main_loss1)
        model = Model(input=[char_vector_input], output=[main_loss])
        return model

    elif model_name == 'BGRU':
        char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
        merged = Masking(mask_value=0., )(char_vector_input)
        x = Bidirectional(GRU(hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        main_loss1 = TimeDistributed(Dense(tag_classes, activation='softplus'))(x)
        main_loss = Activation('softmax')(main_loss1)
        model = Model(input=[char_vector_input], output=[main_loss])
        return model

    elif model_name == 'SimpleRNN':
        char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
        merged = Masking(mask_value=0., )(char_vector_input)
        x = Bidirectional(SimpleRNN(hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        main_loss1 = TimeDistributed(Dense(tag_classes, activation='softplus'))(x)
        main_loss = Activation('softmax')(main_loss1)
        model = Model(input=[char_vector_input], output=[main_loss])
        return model
