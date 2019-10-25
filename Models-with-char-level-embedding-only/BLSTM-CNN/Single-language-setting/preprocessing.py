from __future__ import print_function
import os, io, pickle as cPickle, re, ntpath
import numpy as np, scipy
import gc, sys, getopt

processed_data_path = '../processed_data/'
# encoding = 'ISO-8859-1'
Encoding = 'utf-8'

class Preprocessing:
    def __init__(self, file_name_with_full_path, maximum_word_length = 25):
        self.file_name_with_full_path = file_name_with_full_path
        self.maximum_word_length = maximum_word_length

    def build_char_dic(self):
        char_dic_file = self.file_name_with_full_path + '_char_dictionary'
        if os.path.isfile(char_dic_file):
            return

        dump_char_dic_file = open(char_dic_file, "wb")
        char_dic = {}
        value = 1

        flag = 0
        with io.open(self.file_name_with_full_path, 'r', encoding=Encoding) as f1:
            input_lines = f1.readlines()
        f1.close()

        for line in input_lines:
            if line == '\n':
                continue
            line = line.strip()
            fields = line.split('\t')
            word = fields[0]

            for ch in word:
                if char_dic.get(ch, None) is None:
                    char_dic[ch] = value
                    value += 1

        cPickle.dump(char_dic, dump_char_dic_file, protocol=cPickle.HIGHEST_PROTOCOL)
        dump_char_dic_file.close()

    def get_char_dic(self):
        if not os.path.isfile(self.file_name_with_full_path + '_char_dictionary'):
            print ('in get_char_dic() of preprocessing.py : character dictionary is not found')
            return

        dump_char_dic_file = open( self.file_name_with_full_path + '_char_dictionary', "rb")
        char_dic = cPickle.load(dump_char_dic_file)
        dump_char_dic_file.close()
        return char_dic

    def generate_char_one_hot_vec_and_num_encoding_for_single_word(self, word):
        char_dic = self.get_char_dic()

        word_length = 0
        char_num_array = np.array([])

        for ch in word:
            word_length += 1
            if(word_length > self.maximum_word_length):
                break
            if char_dic.get(ch, None) is not None:
                char_num_array = np.append(char_num_array,int(char_dic[ch]))
            else:
                char_num_array = np.append(char_num_array,0)

        if(word_length < self.maximum_word_length):
            while(word_length != self.maximum_word_length):
                char_num_array = np.append(char_num_array, 0)
                word_length = word_length+1

        return char_num_array

    def generate_char_one_hot_vec_and_num_encoding_for_file(self, file, divide_file_factor=2000, where_to_dump='./'):

        file_substring2 = str(file) + "_char_num_encoded"

        with io.open(file, 'r', encoding="utf-8") as f1:
            input_lines = f1.readlines()
        f1.close()

        input_lines1 = [line for line in input_lines if line.strip() != '']
        input_lines = input_lines1
        input_lines1 = None
        del input_lines1

        all_char_num_vec_array = np.array([])
        count = 0;
        k = 0

        for line in input_lines:
            if line == '\n':
                continue
            while '\t\t' in line:
                line = line.replace('\t\t', '\t')
            line = line.strip()

            fields = line.split('\t')
            word = fields[0]
            char_num_array = self.generate_char_one_hot_vec_and_num_encoding_for_single_word(word)
            all_char_num_vec_array = np.append(all_char_num_vec_array, char_num_array)

            if (count % divide_file_factor == 0 and count != 0) or count == len(input_lines) - 1:
                if os.path.isfile(file_substring2 + '_' + str(k)) is False:
                    all_char_num_vec_array = self.generate_file_and_reset_array(file_substring2, k, all_char_num_vec_array)
                    k += 1
                else:
                    all_char_num_vec_array = np.array([])
                    k += 1

            count += 1

    def generate_class_annotation(self, divide_file_factor=2000):
        with io.open(self.file_name_with_full_path, 'r', encoding=Encoding) as f1:
            input_lines = f1.readlines()
        f1.close()

        file_substring = self.file_name_with_full_path + '_class_annotated'

        class_dic = {}
        class_annotation = []
        value = 1

        for line in input_lines:
            if line == '\n':
                continue
            line = line.strip()
            fields = line.split('\t')
            tag = fields[1]
            if class_dic.get(tag, None) is None:
                class_dic[tag] = value
                value += 1
            class_annotation.append(class_dic[tag])

        all_class_annotation_matrix = np.array([])
        k = 0
        for i in range(len(class_annotation)):
            class_annotation_array = np.zeros(max(class_annotation), dtype=np.int)
            class_annotation_array[class_annotation[i] - 1] = 1
            all_class_annotation_matrix = np.append(all_class_annotation_matrix, class_annotation_array)

            if (i % divide_file_factor == 0 and i != 0) or i == len(class_annotation) - 1:
                if os.path.isfile(file_substring + '_' + str(k)) is False:
                    all_class_annotation_matrix = self.generate_file_and_reset_array(file_substring, k, all_class_annotation_matrix)
                    k += 1
                else:
                    all_class_annotation_matrix = np.array([])
                    k += 1

        reverse_dic = {v: k for k, v in class_dic.items()}
        path = self.file_name_with_full_path + '_class_to_tag_mapping'
        reverse_dic_file = open(path, "wb")
        cPickle.dump(reverse_dic, reverse_dic_file, protocol=cPickle.HIGHEST_PROTOCOL)
        reverse_dic_file.close()
        return max(class_annotation)

    def generate_file_and_reset_array(self, file_substring, sr_num, all_vec_array):
        f_name = file_substring + "_" + str(sr_num)
        tagFile = open(f_name, "wb")
        cPickle.dump(all_vec_array, tagFile, protocol=cPickle.HIGHEST_PROTOCOL)
        tagFile.close()
        print('dumped arrays to file : ', f_name)
        return np.array([])

    def pad_data(self, array, dictionary, file_substring, divide_file_factor = 2000):
       i = 0; j = 0; count = 0; k = 0
       file_substring += '_padded'

       if len(array.shape) != 2:
            print ('error in shape of array: in padding() function of load_data module')
            return None

       zero = np.zeros(int(array.shape[1]))
       store = np.array([])

       for i in range(0, len(array)):
           temp = array[i]
           temp = temp.reshape(len(temp))
           store = np.append(store, temp)
           if (count % divide_file_factor == 0 and count != 0) or i == len(array)-1:
               if os.path.isfile(file_substring + '_' + str(k)) is False:
                   store = self.generate_file_and_reset_array(file_substring, k, store)
                   k += 1
               else:
                   store = np.array([])
                   k += 1

           count += 1

           if dictionary.get(i, None) is not None:
               value = int(dictionary.get(i))
               for j in range(0, value):
                   store = np.append(store, zero)
                   if (count % divide_file_factor == 0 and count != 0) or i == len(array)-1:
                       if os.path.isfile(file_substring + '_' + str(k)) is False:
                           store = self.generate_file_and_reset_array(file_substring, k, store)
                           k += 1
                       else:
                           store = np.array([])
                           k += 1

                   count += 1











