encoding='unicode'
import os, io, cPickle, ntpath, gc, codecs
import preprocessing
# import edit_tree
import numpy as np
Encoding = 'utf-8'



class Load_data:

    def __init__(self, train_file, max_word_length = 25, divide_file_factor = 2000):
        self.train_file = train_file
        self.max_word_length = max_word_length
        self.divide_file_factor = divide_file_factor

    def load_train_char_data(self):
        Preprocessing = preprocessing.Preprocessing(self.train_file, self.max_word_length)
        Preprocessing.build_char_dic()
        Preprocessing.generate_char_one_hot_vec_and_num_encoding_for_file(self.train_file, self.divide_file_factor)
        train_file_substring2 = str(self.train_file) + '_char_num_encoded'
        train_char_data = self.read_parts_from_file(train_file_substring2, self.max_word_length)
        return train_char_data

    def load_test_char_data(self, test_file):
        Preprocessing = preprocessing.Preprocessing(self.train_file, self.max_word_length)
        Preprocessing.generate_char_one_hot_vec_and_num_encoding_for_file(test_file, self.divide_file_factor)
        test_file_substring = test_file + "_char_num_encoded"
        test_char_data = self.read_parts_from_file(test_file_substring, self.max_word_length)
        return test_char_data

    def load_class_annotation_from_train_data(self):
        Preprocessing = preprocessing.Preprocessing(self.train_file, self.max_word_length)
        no_of_classes_1, no_of_classes_2, no_of_classes_3 = Preprocessing.generate_class_annotation(self.divide_file_factor)
        class_annotation_file_substring_1 = self.train_file + '_class_annotated_1'
        class_annotation_file_substring_2 = self.train_file + '_class_annotated_2'
        class_annotation_file_substring_3 = self.train_file + '_class_annotated_3'
        train_data_class_annotation_1 = self.read_parts_from_file(class_annotation_file_substring_1, no_of_classes_1)
        train_data_class_annotation_2 = self.read_parts_from_file(class_annotation_file_substring_2, no_of_classes_2)
        train_data_class_annotation_3 = self.read_parts_from_file(class_annotation_file_substring_3, no_of_classes_3)
        return train_data_class_annotation_1, train_data_class_annotation_2, train_data_class_annotation_3

    def zero_padding_information(self, file, max_sentence_length_in_train_data = None):
        i = 0
        sentences_lengths = []
        with io.open(file, 'r', encoding="utf-8") as f:
            for line in f:
                if line == '\n':
                    sentences_lengths.append(i)
                    i = 0
                else:
                    i += 1
        f.close()

        max_sentence_length = max(sentences_lengths)

        #********* Newly Added **********#
        temp_sentences_lengths = []
        if max_sentence_length_in_train_data is not None:
            if max_sentence_length > max_sentence_length_in_train_data:
                for i in sentences_lengths:
                    if i <= max_sentence_length_in_train_data:
                        temp_sentences_lengths.append(i)
                    else:
                        while i > max_sentence_length_in_train_data:
                            i -= max_sentence_length_in_train_data
                            temp_sentences_lengths.append(max_sentence_length_in_train_data)
                        if i > 0:
                            temp_sentences_lengths.append(i)


        if len(temp_sentences_lengths) > 0:
            sentences_lengths = temp_sentences_lengths
        if max_sentence_length_in_train_data is not None:
            max_sentence_length = max_sentence_length_in_train_data

        indices = []
        temp_list = None
        for i in range(1, len(sentences_lengths)+1):
            temp_list = sentences_lengths[0:i]
            indices.append(sum(temp_list)-1)
        del temp_list

        append_values = [max_sentence_length-i for i in sentences_lengths]
        return dict(zip(indices, append_values)), int(max_sentence_length)

    # def padding(self, array, dictionary, file):
    #     i = 0; j = 0
    #
    #     if len(array.shape) != 2:
    #         print 'error in shape of array: in padding() function of load_data module'
    #         return None
    #     if os.path.isfile(file + '_padded'):
    #         f = open(file + '_padded', 'rb')
    #         store = cPickle.load(f)
    #         f.close()
    #         return store
    #
    #     zero = np.zeros((1, int(array.shape[1])))
    #     store = np.empty((0, int(array.shape[1])), dtype='float32')
    #
    #     for i in range(0, len(array)):
    #         temp = array[i]
    #         temp = temp.reshape(1, len(temp))
    #         store = np.append(store, temp, axis=0)
    #         if dictionary.get(i, None) is not None:
    #             value = int(dictionary.get(i))
    #             for j in range(0, value):
    #                 store = np.append(store, zero, axis=0)
    #
    #     file += '_padded'
    #     f = open(file, 'wb')
    #     cPickle.dump(store, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #     f.close()
    #     return store

    def read_parts_from_file(self, file_name_substring, reshape_size):
        filelist = []
        for root, dirs, files in os.walk('./'):
            for filen in files:
                if file_name_substring in filen:
                    filelist.append(filen)
        f_number = 0
        filelist_in_order = []
        for f in filelist:
            fname = file_name_substring + "_" + str(f_number)
            filelist_in_order.append(fname)
            f_number += 1
        complete_array = np.array([])
        first_access = True
        for fname in filelist_in_order:
            f = open(fname, 'rb')
            complete_array_part = cPickle.load(f)
            f.close()
            complete_array_part = complete_array_part.reshape(-1, reshape_size)
            if (first_access):
                complete_array = complete_array_part
                first_access = False
            else:
                complete_array = np.concatenate((complete_array, complete_array_part), axis=0)
        return complete_array

    def load_padded_data(self, reshape_size, file_substring):
        file_substring += '_padded'
        padded_array = self.read_parts_from_file(file_substring, reshape_size)
        return padded_array




