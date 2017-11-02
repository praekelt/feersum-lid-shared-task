# feersum-lid-shared-task
This git repository hosts the data and code that accompanies a paper 'Improved Text Language Identification for the South African Languages'
published at the Prasa/RobMech 2017 conference.

The paper is on text language identification of short pieces of text of around 15 characters. Hopefully this repo will be useful to create a text language
identification shared task for South African languages. Example training and testing data used in this paper is in the folder lid_task_2017a.

## The Paper
Please cite our paper or attribute the use of the data to:

B. Duvenhage, M. Ntini and P. Ramonyai, "Improved Text Language Identification for the South African Languages," The 28th Annual Symposium
of the Pattern Recognition Association of South Africa, 2017.

A preprint of the paper is available at https://arxiv.org/abs/1711.00247

## The Data
The data is from the NHCLT Text Corpera collected by South African Department of Arts and Culture & Centre for Text Technology (CTexT,
North-West University, South Africa). Each folder in the data contains the original and improved text corpera for a single language.

The corpera is improved from the original version as explained in the paper. The data was cleaned and incorrectly labelled samples were
relabelled.

The data may be loaded and split into training and testing sets by using text_classifier.load_sentences_all() from text_classifier.py.

Folder folder lid_task_2017a contains example labelled training and testing sets with a 3000:1000 split and full sentence lengths between
200 and 300 characters are given in train_full_3k.csv and test_full_1k.csv An example test set with truncated text length of approximately
15 characters is given in test_15_1k.csv. The sample files have one header line and the format is lang_id, text.

## The Code
The code hosted here may be used to regenerate the results in the paper.

train_baseline.py is required to train the baseline Naive Bayesian LID classifier.

lang_ident_test.py is used to test the baseline and baseline+lexicon LID. The lexicon is derived from the improved NHCLT Text Corpera.

### If you are new to python then you can do the following from a terminal to get started:
`git clone https://github.com/praekelt/feersum-lid-shared-task.git`

`cd feersum-lid-shared-task`

`virtualenv -p /usr/local/bin/python3.5 .pyenv`
 (alternatively use python3.6)

`source .pyenv/bin/activate`

`pip install pip-tools`

`pip install appdirs`

`pip install sklearn`

`pip install numpy`

`pip install scipy`

### If you don't have virtualenv installed first run:
`pip install virtualenv`

`sudo /usr/bin/easy_install virtualenv`

### If you don't have pip or python:
First install Python 3.5 or 3.6.

### To run the code to train the baseline classifier:
`cd code`

`python train_baseline.py`

### To test LID of 15 char strings with the lexicon:
`python lang_ident_test.py`



