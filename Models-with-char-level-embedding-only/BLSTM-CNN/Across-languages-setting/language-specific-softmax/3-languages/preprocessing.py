from __future__ import print_function
import os, io, cPickle, re, ntpath
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

        file_substring_1 = self.file_name_with_full_path + '_class_annotated_1'
        file_substring_2 = self.file_name_with_full_path + '_class_annotated_2'
        file_substring_3 = self.file_name_with_full_path + '_class_annotated_3'

        class_dic_1 = {}
        class_annotation_1 = []
        value_1 = 1

        class_dic_2 = {}
        class_annotation_2 = []
        value_2 = 1

        class_dic_3 = {}
        class_annotation_3 = []
        value_3 = 1

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

        for line in input_lines:
            if line == '\n':
                continue
            line = line.strip()
            fields = line.split('\t')
            word = fields[0]
            tag = fields[1]

            if lan_id_dic[word[0:1]] == 1:
                if class_dic_1.get(tag, None) is None:
                    class_dic_1[tag] = value_1
                    value_1 += 1
                class_annotation_1.append(class_dic_1[tag])
                class_annotation_2.append(0)
                class_annotation_3.append(0)
            elif lan_id_dic[word[0:1]] == 2:
                if class_dic_2.get(tag, None) is None:
                    class_dic_2[tag] = value_2
                    value_2 += 1
                class_annotation_1.append(0)
                class_annotation_2.append(class_dic_2[tag])
                class_annotation_3.append(0)
            else:
                if class_dic_3.get(tag, None) is None:
                    class_dic_3[tag] = value_3
                    value_3 += 1
                class_annotation_1.append(0)
                class_annotation_2.append(0)
                class_annotation_3.append(class_dic_3[tag])

        all_class_annotation_matrix_1 = np.array([])
        all_class_annotation_matrix_2 = np.array([])
        all_class_annotation_matrix_3 = np.array([])

        if len(class_annotation_1) != len(class_annotation_2) != len(class_annotation_3):
            print('\n\nError in generate class annotation: preprocessing.py\n\n')

        k = 0
        for i in range(len(class_annotation_1)):
            class_annotation_array_1 = np.zeros(max(class_annotation_1), dtype=np.int)
            class_annotation_array_2 = np.zeros(max(class_annotation_2), dtype=np.int)
            class_annotation_array_3 = np.zeros(max(class_annotation_3), dtype=np.int)

            if class_annotation_1[i] != 0:
                class_annotation_array_1[class_annotation_1[i] - 1] = 1
            elif class_annotation_2[i] != 0:
                class_annotation_array_2[class_annotation_2[i] - 1] = 1
            else:
                class_annotation_array_3[class_annotation_3[i] - 1] = 1

            all_class_annotation_matrix_1 = np.append(all_class_annotation_matrix_1, class_annotation_array_1)
            all_class_annotation_matrix_2 = np.append(all_class_annotation_matrix_2, class_annotation_array_2)
            all_class_annotation_matrix_3 = np.append(all_class_annotation_matrix_3, class_annotation_array_3)

            if (i % divide_file_factor == 0 and i != 0) or i == len(class_annotation_1) - 1:
                if os.path.isfile(file_substring_1 + '_' + str(k)) is False:
                    all_class_annotation_matrix_1 = self.generate_file_and_reset_array(file_substring_1, k, all_class_annotation_matrix_1)

                if os.path.isfile(file_substring_2 + '_' + str(k)) is False:
                    all_class_annotation_matrix_2 = self.generate_file_and_reset_array(file_substring_2, k, all_class_annotation_matrix_2)

                if os.path.isfile(file_substring_3 + '_' + str(k)) is False:
                    all_class_annotation_matrix_3 = self.generate_file_and_reset_array(file_substring_3, k, all_class_annotation_matrix_3)

                all_class_annotation_matrix_1 = np.array([])
                all_class_annotation_matrix_2 = np.array([])
                all_class_annotation_matrix_3 = np.array([])
                k += 1


        reverse_dic_1 = {v: k for k, v in class_dic_1.iteritems()}
        reverse_dic_2 = {v: k for k, v in class_dic_2.iteritems()}
        reverse_dic_3 = {v: k for k, v in class_dic_3.iteritems()}
        path1 = self.file_name_with_full_path + '_class_to_tag_mapping_1'
        path2 = self.file_name_with_full_path + '_class_to_tag_mapping_2'
        path3 = self.file_name_with_full_path + '_class_to_tag_mapping_3'
        reverse_dic_file1 = open(path1, "wb")
        reverse_dic_file2 = open(path2, "wb")
        reverse_dic_file3 = open(path3, "wb")
        cPickle.dump(reverse_dic_1, reverse_dic_file1, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(reverse_dic_2, reverse_dic_file2, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(reverse_dic_3, reverse_dic_file3, protocol=cPickle.HIGHEST_PROTOCOL)
        reverse_dic_file1.close()
        reverse_dic_file2.close()
        reverse_dic_file3.close()
        return max(class_annotation_1), max(class_annotation_2), max(class_annotation_3)



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











