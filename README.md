# Neural-Morphological-Tagging

This repository contains the code of character-level neural morphological tagging for two models - BLSTM-BLSTM and BLSTM-CNN. For each of the models, 3 different learning setups are there. They are as follows.

1. Single-language-setting.
2. Learning-across-keys.
3. Across-languages-setting.

In Single-language-setting, the model is trained and tested on a single language.

In Learning-across-keys setup, the model is trained and tested on a single language, but here multi-task learning is exploited across universal morphological keys such as part-of-speech (POS), case, degree, gender etc.

In Across-languages-setting, multilingual training is done on similar languages.

For each of the learning setups, the following files are present in the corresponding directories.

1. main.py
2. models.py
3. load_data.py
4. preprocessing.py
5. f_measure_accuracy.py
6. parameters

Place your data files i.e. train, dev and test files in the same place where the codes are there.

When you just want to train the model on the training set, run the following command,

python main.py parameters &lt;train-file&gt;

If you want to train the model on the training-set and test it on the test-set in a pipeline, run the following command,
python main.py parameters &lt;train-file&gt; &lt;test-file&gt;

The tagging output for the &lt;test-file&gt; will be &lt;test-file&gt;-output.

*********************************************************************************************

parameters file contains the value of the model hyperparameters. For example, in the Single-language-setting of the BLSTM-BLSTM model, the value in the parameters file is as follows.
  
The first line should be the cell type of the character level network. It should be BLSTM.

The second line should be the maximum word length in number of characters. For the words having number of characters more than this value, the excess characters will be truncated out. In our experiments, We set this to 25, but you can vary it with different values to check the model's performance.

The third line should be the number of neurons in the cell of the character level network.

The fourth line should be the cell type of the tag-classification network. It should be BLSTM.

The fifth line should be the number of neurons in the cell of the tag-classification network.

The sixth line should be the number of epochs for training of the network.

The seventh line should be the batch size for training of the network.

The eighth line is the divide file factor. For large size data, creating the data matrices which will be generated while running the code, takes too much time. To reduce the time, we fragment the matrices part-by-part and then dump them in the disk and reload part-by-part. For our experiments we kept this value as 2000. It's user's choice to set this value. Normally set this value in terms of thousands i.e. 1000/2000/3000.

For each of the 3 different learning setups of BLSTM-BLSTM and BLSTM-CNN, the syntax of the parameters files are given in the individual "ReadMe" files in the corresponding directories. Note that for different languages, the values of the hyperparameters should be different i.e. you have to change/tune the values to get the optimum models.

*********************************************************************************************

Training file format: - In the training file, words and their respecive morpholgical tags should be TAB separated. 
After each sentence, there should be a new line gap. 
The format of the train file is like below:

word1&lt;TAB&gt;tag1

word2&lt;TAB&gt;tag2

word3&lt;TAB&gt;tag3

word4&lt;TAB&gt;tag4
 
 




word5&lt;TAB&gt;tag5

word6&lt;TAB&gt;tag6

word7&lt;TAB&gt;tag7

word8&lt;TAB&gt;tag8

word9&lt;TAB&gt;tag9

An example sentence from Hindi is shown here.

इसके	PRON|Case=Acc,Gen|Number=Sing|Person=3|Poss=Yes|PronType=Prs

अतिरिक्त	ADP|AdpType=Post

गुग्गुल	PROPN|Case=Nom|Gender=Masc|Number=Sing|Person=3

कुंड	PROPN|Case=Nom|Gender=Masc|Number=Sing|Person=3

,	PUNCT

भीम	PROPN|Case=Nom|Gender=Masc|Number=Sing|Person=3

गुफा	PROPN|Case=Nom|Gender=Fem|Number=Sing|Person=3

तथा	CCONJ

भीमशिला	PROPN|Case=Nom|Gender=Fem|Number=Sing|Person=3

भी	PART

दर्शनीय	ADJ|Case=Nom

स्थल	NOUN|Case=Nom|Gender=Masc|Number=Plur|Person=3

हैं	AUX|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act

।	PUNCT

Note that in our experiments we use universal dependencies datasets. As a preprocessing, we join the 4th and the 6th columns together which represents the tag of a word.

Test file and development file format:


In the test/dev file, after each sentence, there should be a new line gap. The format is like below:

Case 1. If you want to evaluate the trained model's performance against the test/dev set, then the format of the test/dev file is same as that of the train file i.e words and their respective gold tags are TAB separated and after each sentence there is a new line gap.

Case 2. If you just want to get the tagged output for the test set and don't want to evaluate the trained model on the test set, then keep the words aligned per line. After each sentence, there should be a new line gap. The format is like below:

word1

word2

word3

word4

word5


word6

word7

word8

word9

For our experiments, we explored 5 Indic languages from 2 different families - 1. Indo-Aryan (Hindi, Marathi, Sanskrit) 2. Dravidian (Tamil and Telugu). Additionally we included 2 other severely resource-scarce languages namely Coptic and Kurmanji. The codes given here are generic so that they can be used for training and testing on any arbitrary language.

# Requirements:
======================

Python 2.7

Keras 1

(However, to run the codes for Python 3 and Keras 2, name of some functions should be changed).




# Citation
======================

If you use the codes, please cite the following paper.

"NeuMorph: Neural Morphological Tagging for Low-Resource Languages - An Experimental Study for Indic Languages" - ABHISEK CHAKRABARTY, AKSHAY CHATURVEDI, UTPAL GARAIN. DOI.https://10.1145/3342354
