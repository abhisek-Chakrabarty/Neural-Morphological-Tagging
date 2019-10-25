from __future__ import print_function
import load_data, models, preprocessing, f_measure_accuracy
import os, io, cPickle, re, ntpath
import numpy as np
import gc, sys
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
nb_epoch = int(input_lines[3].strip())
batch_size = int(input_lines[4].strip())
no_of_kernels = int(input_lines[5].strip())
filter_window_length = int(input_lines[6].strip())
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

train_data_class_annotation_array, label_dictionary = ld.load_class_annotation_from_train_data()
print ('train_data_class_annotation:\n====================')
i = 0; j = 0
for i in label_dictionary:
    print ('for label \'', i, '\' class annoattion shape is: ', str(train_data_class_annotation_array[j].shape))
    j += 1

#***************************************************************************************#
#*************************** Padding of Class Annotation Data **************************#
 
i = 0; j = 0
for i in label_dictionary:
    preprocessing.Preprocessing(train_file).pad_data(train_data_class_annotation_array[j], padding_dictionary, train_file + '_class_annotation_' + i, divide_file_factor)
    j += 1
 
train_data_class_annotation_array = []
gc.collect()
i = 0; j = 0
 
for i in label_dictionary:
    train_data_class_annotation = ld.load_padded_data(len(label_dictionary[i]), train_file + '_class_annotation_' + i)
    train_data_class_annotation_array.append(train_data_class_annotation)
    print('After Padding train_data_class_annotation.shape for label \'' + i + '\' is : ' + str(train_data_class_annotation.shape))
    train_data_class_annotation = None
    gc.collect()
train_char_vectors = train_char_vectors.astype('float32')

#********************************** Reshaping *****************************************#

train_char_vectors = train_char_vectors.reshape(-1, max_sentence_length, int(train_char_vectors.shape[1]))
i = 0; j = 0
for i in label_dictionary:
    train_data_class_annotation_array[j] = train_data_class_annotation_array[j].reshape(-1, max_sentence_length, int(train_data_class_annotation_array[j].shape[1]))
    j += 1
 
zeros = np.zeros((train_char_vectors.shape[0],1,train_char_vectors.shape[2]), dtype='float32')
 
for i in range(0, filter_window_length):
    train_char_vectors = np.hstack((zeros, train_char_vectors))
    train_char_vectors = np.hstack((train_char_vectors, zeros))
 
del zeros
gc.collect()

#**************************************************************************************#

no_of_classes_per_label = []
name_of_labels = []
for i in label_dictionary:
    no_of_classes_per_label.append(int(len(label_dictionary[i])))
    name_of_labels.append(i)

tag_classify_model = models.get_tag_classify_model_with_CNN(max_sentence_length+2*filter_window_length, 2*char_feature_output, no_of_kernels, filter_window_length, no_of_classes_per_label)
print ('tree classify model summary: ')
print (tag_classify_model.summary())

tag_classify_model.compile(optimizer='Nadam', loss='mean_squared_logarithmic_error',metrics=['accuracy'])
tag_classify_model_file_name = train_file + '_tag_classify_with_CNN.h5'
if os.path.isfile(os.path.join('./', tag_classify_model_file_name)):
    tag_classify_model.load_weights(os.path.join('./', tag_classify_model_file_name))
    print ('loaded ' + tag_classify_model_file_name + 'from disk')
else:
    tag_classify_model.fit([train_char_vectors], train_data_class_annotation_array, nb_epoch=nb_epoch, batch_size=batch_size, )
    print('training completed')
    tag_classify_model.save_weights(os.path.join('./', tag_classify_model_file_name))
    print(tag_classify_model_file_name + ' : saved weights in the disk')

del train_char_vectors, train_data_class_annotation_array
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
new_test_char_vectors = test_char_vectors
zeros = np.zeros((test_char_vectors.shape[0],1,test_char_vectors.shape[2]), dtype='float32')
for i in range(0, filter_window_length):
    test_char_vectors = np.hstack((zeros, test_char_vectors))
    test_char_vectors = np.hstack((test_char_vectors, zeros))
del zeros
gc.collect()
#************************************************************************************#
#******************************** Prediction ****************************************#

output = tag_classify_model.predict([test_char_vectors])
for i in range(0, len(no_of_classes_per_label)):
    output[i] = output[i].reshape(int(output[i].shape[0])*int(output[i].shape[1]), -1)
new_test_char_vectors = new_test_char_vectors.reshape(int(new_test_char_vectors.shape[0])*int(new_test_char_vectors.shape[1]), -1)
flag = [True if new_test_char_vectors[i].any() else False for i in range(0, len(new_test_char_vectors))]
del test_char_vectors, new_test_char_vectors
gc.collect()


output_tag_classes_label_wise = []

for i in range(0, len(output)):
    output_tag_classes = []
    for j in range(0, len(output[i])):
        if flag[j] == False:
            continue
        max_prob = -1.0; index = -1
        for k in range(0, len(output[i][j])):
            if max_prob < output[i][j][k]:
                max_prob = output[i][j][k]
                index = k
        output_tag_classes.append(index)
    output_tag_classes_label_wise.append(output_tag_classes)
    del output_tag_classes
gc.collect()


for i in label_dictionary:
    label_dictionary[i] = list(enumerate(label_dictionary[i]))

i = 0
total_evaluation_correct_count = 0.0
evaluation_dictionary_for_labels = {}
temporary_array = [0.0, 0.0]
for j in name_of_labels:
    evaluation_dictionary_for_labels[j] = temporary_array


output_file = str(test_file) + '_output'
file_writer = io.open(output_file, 'w', encoding="utf-8")
l = 0.0
to_evaluate = False
with io.open(test_file, 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == '':
            file_writer.write(u'\n')
            continue
        fields = line.split('\t')

        string = ''
        total_evaluation_flag = True
        if len(fields) >= 2:
            fields[1] = 'POS=' + fields[1]
            to_evaluate = True

        for j in range(0, len(output_tag_classes_label_wise)):
            string += name_of_labels[j] + '=' + label_dictionary[name_of_labels[j]][output_tag_classes_label_wise[j][i]][1] + '|'

            if len(fields) >= 2:
                gold_labels_and_tags = fields[1].split('|')

                for k in range(0, len(gold_labels_and_tags)):
                    index = gold_labels_and_tags[k].index('=')
                    gold_label = gold_labels_and_tags[k][0:index]
                    if gold_label == name_of_labels[j]:
                        gold_tag = gold_labels_and_tags[k][index+1:len(gold_labels_and_tags[k])]
                        if gold_tag != 'NULL':
                            temp_list = list(evaluation_dictionary_for_labels[gold_label])
                            temp_list[0] += 1
                            del evaluation_dictionary_for_labels[gold_label]
                            evaluation_dictionary_for_labels[gold_label] = temp_list
                            if gold_tag != label_dictionary[name_of_labels[j]][output_tag_classes_label_wise[j][i]][1]:
                                total_evaluation_flag = False
                            else:
                                temp_list = list(evaluation_dictionary_for_labels[gold_label])
                                temp_list[1] += 1
                                del evaluation_dictionary_for_labels[gold_label]
                                evaluation_dictionary_for_labels[gold_label] = temp_list
        if total_evaluation_flag is True:
            total_evaluation_correct_count += 1
        string = string[0:len(string)-1]
        i += 1
        file_writer.write(fields[0] + '\t' + string + '\n')
        if to_evaluate == True:
            l += f_measure_accuracy.f_measure_acc_for_single_word(str(fields[1]), str(string))
f.close()
file_writer.close()

if to_evaluate is True:
    print ('total_accuracy = ' + str((100 * total_evaluation_correct_count)/i) + '\%')
    print ('f-score = ', float(l)*100.0/i)
    for i in evaluation_dictionary_for_labels:
        if evaluation_dictionary_for_labels[i][0] == 0:
            print ('\'' + i + '\' accuracy = 0%')
        else:
            print ('\'' + i + '\' accuracy = ', str((100*evaluation_dictionary_for_labels[i][1])/float(evaluation_dictionary_for_labels[i][0])) + '\%')
    print ('evaluation_dictionary_for_labels' + '\n' + str(evaluation_dictionary_for_labels))

#####################################################################
# i = 0
# output_file = str(test_file) + '_output'
# file_writer = io.open(output_file, 'w', encoding="utf-8")
# with io.open(test_file, 'r', encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if line == '':
#             file_writer.write(u'\n')
#             continue
#         fields = line.split('\t')
#
#         string = ''
#         for j in range(0, len(output_tag_classes_label_wise)):
#             string += name_of_labels[j] + '=' + label_dictionary[name_of_labels[j]][output_tag_classes_label_wise[j][i]][1] + '|'
#         string = string[0:len(string)-1]
#         i += 1
#         file_writer.write(fields[0] + '\t' + string + '\n')
# f.close()
# file_writer.close()
######################################################################











