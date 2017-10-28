# feersum-lid-shared-task
This git repository hosts the data and code that accompanies a paper 'Improved Text Language Identification for the South African Languages'
published at the Prasa/RobMech 2017 conference.

The paper is on text language identification of short pieces of text of around 15 characters. Hopefully this repo will be useful to create a text language
identification shared task for South African languages.

## The Data
The data is from the NHCLT Text Corpera collected by South African Department of Arts and Culture & Centre for Text Technology (CTexT,
North-West University, South Africa). Each folder in the data contains the original and improved text corpera for a single language.

The corpera is improved from the original version as explained in the paper. The data was cleaned and incorrectly labelled samples were
relabelled.

The data may be loaded and split into training and testing sets by using text_classifier.load_sentences_all() from text_classifier.py. Example labelled
training and testing sets with a 3000:1000 split and full sentence lengths between 200 and 300 characters are given in train_full_3k.csv and test_full_1k.csv
An example test set with truncated text length of approximately 15 characters is given in test_15_1k.csv. The file has one header line and the format is
lang_id, text.

## The Code
The code hosted here may be used to regenerate the results in the paper. The paper is still under review and the code will be committed here soon.
