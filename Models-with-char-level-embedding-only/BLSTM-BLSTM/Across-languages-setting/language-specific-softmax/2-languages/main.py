from __future__ import print_function
import load_data, models, preprocessing, f_measure_accuracy
import os, io, cPickle, re, ntpath
import numpy as np
import gc, sys
Encoding = 'utf-8'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

np.random.seed(1337)  # for reproducibility
## Reading the Parameters
with io.open(str(sys.argv[1]), 'r') as f1:
    input_lines = f1.readlines()
f1.close()


char_network_cell = str(input_lines[0].strip())
max_word_length = int(input_lines[1].strip())
char_feature_output = int(input_lines[2].strip())
tag_classify_network_cell = str(input_lines[3].strip())
hidden_size = int(input_lines[4].strip())
nb_epoch = int(input_lines[5].strip())
batch_size = int(input_lines[6].strip())
divide_file_factor = int(input_lines[7].strip())

del input_lines
input_lines = None


# Load Training Char Data

train_file = str(sys.argv[2])
ld = load_data.Load_data(train_file, max_word_length, divide_file_factor)   #give the training file name

train_char_data = ld.load_train_char_data()
print ('train_char_data.shape: ' + str(train_char_data.shape))
embedded_char_vector_length = len(preprocessing.Preprocessing(train_file).get_char_dic())

#***************************** Training Phase Start *****************************#
#********************** Building Char Model *************************#

char_model = models.get_char_model(char_network_cell, max_word_length, embedded_char_vector_length, char_feature_output)
print('char model summary:')
print(char_model.summary())

char_model_file_name = train_file + '_char_level_' + char_network_cell + '.h5'
if os.path.isfile(os.path.join('./', char_model_file_name)):
    char_model.load_weights(os.path.join('./', char_model_file_name))
    print ('loaded ' + char_model_file_name + ' from disk')
else:
    char_model.save_weights(os.path.join('./', char_model_file_name))
    print (char_model_file_name + ' : saved weights in the disk')

train_char_vectors = char_model.predict([train_char_data])
print ('train_char_vectors.shape is: ' + str(train_char_vectors.shape))

del train_char_data
gc.collect()

#*************************** Padding of Training Char Data *****************************#
 
padding_dictionary, max_sentence_length = ld.zero_padding_information(train_file)
print ('max_sentence_length: ', max_sentence_length)
 
preprocessing.Preprocessing(train_file).pad_data(train_char_vectors, padding_dictionary, train_file + '_char_vectors', divide_file_factor)
reshape_size = int(train_char_vectors.shape[1])
del train_char_vectors
gc.collect()
train_char_vectors = ld.load_padded_data(reshape_size, train_file + '_char_vectors')
 
print ('After Padding train_char_vectors.shape is : ' + str(train_char_vectors.shape))
 
#***************************************************************************************#
#*************************** Loading Class Annotated Data ******************************#

train_data_class_annotation_1, train_data_class_annotation_2 = ld.load_class_annotation_from_train_data()
print ('train_data_class_annotation_1.shape: ' + str(train_data_class_annotation_1.shape))
print ('train_data_class_annotation_2.shape: ' + str(train_data_class_annotation_2.shape))
no_of_classes_1 = int(train_data_class_annotation_1.shape[1])
no_of_classes_2 = int(train_data_class_annotation_2.shape[1])

#***************************************************************************************#
#*************************** Padding of Class Annotation Data **************************#
 
preprocessing.Preprocessing(train_file).pad_data(train_data_class_annotation_1, padding_dictionary, train_file + '_class_annotation_1', divide_file_factor)
preprocessing.Preprocessing(train_file).pad_data(train_data_class_annotation_2, padding_dictionary, train_file + '_class_annotation_2', divide_file_factor)
reshape_size_1 = int(train_data_class_annotation_1.shape[1])
reshape_size_2 = int(train_data_class_annotation_2.shape[1])
del train_data_class_annotation_1, train_data_class_annotation_2
gc.collect()
train_data_class_annotation_1 = ld.load_padded_data(reshape_size_1, train_file + '_class_annotation_1')
train_data_class_annotation_2 = ld.load_padded_data(reshape_size_2, train_file + '_class_annotation_2')
print ('After Padding train_data_class_annotation_1.shape is : ' + str(train_data_class_annotation_1.shape))
print ('After Padding train_data_class_annotation_2.shape is : ' + str(train_data_class_annotation_2.shape))
train_char_vectors = train_char_vectors.astype('float32')
 
#********************************** Reshaping *****************************************#
 
train_char_vectors = train_char_vectors.reshape(-1, max_sentence_length, int(train_char_vectors.shape[1]))
train_data_class_annotation_1 = train_data_class_annotation_1.reshape(-1, max_sentence_length, int(train_data_class_annotation_1.shape[1]))
train_data_class_annotation_2 = train_data_class_annotation_2.reshape(-1, max_sentence_length, int(train_data_class_annotation_2.shape[1]))
gc.collect()
 
#**************************************************************************************#

tag_classify_model = models.get_tag_classify_model(tag_classify_network_cell, max_sentence_length, 2*char_feature_output, hidden_size, no_of_classes_1, no_of_classes_2)
print ('tree classify model summary: ')
print (tag_classify_model.summary())

tag_classify_model.compile(optimizer='Nadam', loss='mean_squared_logarithmic_error',metrics=['accuracy'])
tag_classify_model_file_name = train_file + '_tag_classify_' + tag_classify_network_cell + '.h5'
if os.path.isfile(os.path.join('./', tag_classify_model_file_name)):
    tag_classify_model.load_weights(os.path.join('./', tag_classify_model_file_name))
    print ('loaded ' + tag_classify_model_file_name + 'from disk')
    #tag_classify_model.fit([train_char_vectors], [train_data_class_annotation_1, train_data_class_annotation_2], nb_epoch=nb_epoch, batch_size=batch_size, )
    #tag_classify_model.save_weights(os.path.join('./', tag_classify_model_file_name))
else:
    tag_classify_model.fit([train_char_vectors], [train_data_class_annotation_1, train_data_class_annotation_2], nb_epoch=nb_epoch, batch_size=batch_size, )
    print('training completed')
    tag_classify_model.save_weights(os.path.join('./', tag_classify_model_file_name))
    print(tag_classify_model_file_name + ' : saved weights in the disk')

del train_char_vectors, train_data_class_annotation_1, train_data_class_annotation_2
gc.collect()

#************************************ Preprocessing of Test File *********************#

test_file = None
if len(sys.argv) > 3:
    test_file = str(sys.argv[3])
else:
    sys.exit(0)

test_char_data = ld.load_test_char_data(test_file)
test_char_vectors = char_model.predict([test_char_data])
del test_char_data
gc.collect()

#*************************** Padding of Test Files **********************************#

print ('Before padding test char vectors shape ', str(test_char_vectors.shape))
padding_dictionary, _ = ld.zero_padding_information(test_file, max_sentence_length)
preprocessing.Preprocessing(train_file).pad_data(test_char_vectors, padding_dictionary, test_file + '_as_test_and_' + train_file + '_as_train_char_vectors', divide_file_factor)
reshape_size = int(test_char_vectors.shape[1])
del test_char_vectors
gc.collect()
test_char_vectors = ld.load_padded_data(reshape_size, test_file + '_as_test_and_' + train_file + '_as_train_char_vectors')
print ('After padding test char vectors shape ', str(test_char_vectors.shape))
test_char_vectors = test_char_vectors.astype('float32')

#*************************** Reshaping of Test Data Arrays **************************#

test_char_vectors = test_char_vectors.reshape(-1, max_sentence_length, int(test_char_vectors.shape[1]))

#******************************** Prediction ****************************************#

output_1, output_2 = tag_classify_model.predict([test_char_vectors])
output_1 = output_1.reshape(int(output_1.shape[0])*int(output_1.shape[1]), -1)
output_2 = output_2.reshape(int(output_2.shape[0])*int(output_2.shape[1]), -1)
test_char_vectors = test_char_vectors.reshape(int(test_char_vectors.shape[0])*int(test_char_vectors.shape[1]), -1)
flag = [True if test_char_vectors[i].any() else False for i in range(0, len(test_char_vectors))]


with io.open(train_file, 'r', encoding=Encoding) as f1:
    input_lines = f1.readlines()
f1.close()
lan_id_dic = {}
id = 1
for line in input_lines:
    if line == '\n':
        continue
    line = line.strip()
    fields = line.split('\t')
    word = fields[0]
    char = word[0:1]
    if lan_id_dic.get(char, None) is None:
        lan_id_dic[char] = id
        id += 1
input_lines = None
gc.collect()

output_tag_classes_1 = []
output_tag_classes_2 = []
for i in range(0, len(test_char_vectors)):
    if flag[i] == False:
        continue
    max_prob = -1.0; index = -1
    for j in range(0, len(output_1[i])):
        if max_prob < output_1[i][j]:
            max_prob = output_1[i][j]
            index = j
    output_tag_classes_1.append(index)

    max_prob = -1.0; index = -1
    for j in range(0, len(output_2[i])):
        if max_prob < output_2[i][j]:
            max_prob = output_2[i][j]
            index = j
    output_tag_classes_2.append(index)


i = 0; j = 0; k = 0.0; evaluation_flag = 0


class_to_tag_dictionary_file_1 = open(train_file + '_class_to_tag_mapping_1', "rb")
class_to_tag_dictionary_file_2 = open(train_file + '_class_to_tag_mapping_2', "rb")
class_to_tag_dictionary_1 = cPickle.load(class_to_tag_dictionary_file_1)
class_to_tag_dictionary_2 = cPickle.load(class_to_tag_dictionary_file_2)
class_to_tag_dictionary_file_1.close()
class_to_tag_dictionary_file_2.close()

output_file = str(test_file) + '_output'
file_writer = io.open(output_file, 'w', encoding="utf-8")
with io.open(test_file, 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == '':
            file_writer.write(u'\n')
            continue
        fields = line.split('\t')
        word = fields[0]
        char = word[0:1]
        if lan_id_dic[char] == 1:
            predicted_tag = class_to_tag_dictionary_1[int(output_tag_classes_1[i])+1]
        else:
            predicted_tag = class_to_tag_dictionary_2[int(output_tag_classes_2[i]) + 1]
        i += 1
        if len(fields) >= 2:
            evaluation_flag = 1
        if evaluation_flag == 1:
            if fields[1] == predicted_tag:
                j += 1;
            k += f_measure_accuracy.f_measure_acc_for_single_word(str(fields[1]), str(predicted_tag))
        file_writer.write(fields[0] + '\t' + predicted_tag + '\n')
f.close()
file_writer.close()
if evaluation_flag == 1:
    print ('Accuracy = ', float(j)*100.0/i)
    print ('f-score = ', float(k)*100.0/i)






















