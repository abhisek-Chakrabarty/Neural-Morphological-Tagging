# Neural-Morphological-Tagging
===============================
This repository contains the code of character-level neural morphological tagging for two models - BLSTM-BLSTM and BLSTM-CNN. For each of the models, 3 different learning setups are there. They are as follows.

1. Single-language-setting.
2. Learning-across-keys.
3. Across-languages-setting.

In Single-language-setting, the model is trained and tested on a single language.

In Learning-across-keys setup, the model is trained and tested on a single language, but here multi-task learning is exploited across universal morphological keys such as part-of-speech (POS), case, degree, gender etc.

In Across-languages-setting, multilingual training is done on similar languages.

For each of the learning setups, the following files are present in the corresponding directories. Place your data files i.e. train, dev and test files in the same place where the codes are there.

1. main.py
2. models.py
3. load_data.py
4. preprocessing.py
5. f_measure_accuracy.py
6. parameters

When you just want to train the model on the training set, run the following command,
python main.py parameters train-file

If you want to train the model on the training-set and test it on the test-set in a pipeline, run the following command,
python main.py parameters train-file test-file

In the parameters file, the value of the hyperparameters should be there.

Training file format: - In the training file, words and their respecive morpholgical tags should be tab separated. 
After each sentence, there should be a new line gap. 
The format of the train file is like below:

word1	tag1
word2	tag2
word3	tag3
word4	tag4

word5	tag5
word6	tag6
word7	tag7
word8	tag8
word9	tag9

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

Test file and development file format: In the test/dev file, after each sentence, there should be a new line gap. The format is like below:

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
