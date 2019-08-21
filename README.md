# Neural-Morphological-Tagging
This repository contains the code of character-level neural morphological tagging for two models - BLSTM-BLSTM and BLSTM-CNN. For each of the models, 3 different learning setups are there. They are as follows.

1. Single-language-setting.
2. Learning-across-keys.
3. Across-languages-setting.

In Single-language-setting, the model is trained and tested on a single language.

In Learning-across-keys setup, the model is trained and tested on a single language, but here multi-task learning is exploited across universal morphological keys such as part-of-speech (POS), case, degree, gender etc.

In Across-languages-setting, multilingual training is done on similar languages.

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
