# feersum-lid-shared-task
This git repository hosts the data and code that accompanies 'Improved Text Language Identification for the South African Languages'
published at the Prasa/RobMech 2017 conference as well as 'Short Text Language Identification for Under Resourced Languages' published at NeurIPS 2019 ML4D workshop.

The papers are on text language identification of short pieces of text of around 15 characters. Hopefully this repo can be useful to create a text language identification shared task for South African languages. Example training and testing data used in the 2017 paper is in the folder lid_task_2017a.

## The Papers
Please cite our papers or attribute the use of the lid_task_2017a data to:

B. Duvenhage, M. Ntini and P. Ramonyai, "Improved Text Language Identification for the South African Languages," The 28th Annual Symposium of the Pattern Recognition Association of South Africa, 2017. Available at https://arxiv.org/abs/1711.00247

B. Duvenhage, "Short Text Language Identification for Under Resourced Languages," The NeurIPS 2018 Workshop on Machine Learning for the Developing World. Available at https://arxiv.org/abs/1911.07555


## The Data
The texts in the 'data' folder is from the NCHLT Text Corpora collected by South African Department of Arts and Culture & Centre for Text Technology (CTexT,
North-West University, South Africa). Each folder in the data contains the original and improved text corpora for a single language.

The corpora is improved from the original version as explained in the 2017 paper. The data was manually inspected and incorrectly labelled samples were
relabelled. Folder lid_task_2017a contains example labelled training and testing sets with a 3000:1000 split and full sentence lengths between
200 and 300 characters are given in train_full_3k.csv and test_full_1k.csv An example test set with truncated text length of approximately
15 characters is given in test_15_1k.csv. The sample files have one header line and the format is lang_id, text.

## The Code
The code hosted here may be used to regenerate the results in the papers.

### 'code_neurips_2019' folder:
Run 'jupyter notebook', open the NeurIPS notebook and run all the cells. 

### 'code' folder:
train_baseline.py is required to train the baseline Naive Bayesian LID classifier for the 2017 paper.

lang_ident_test.py is used to test the baseline and baseline+lexicon LID. The lexicon is derived from the improved NCHLT Text Corpora.

To run the code to train the baseline classifier: `cd code` & `python train_baseline.py`

To test LID of 15 char strings with the lexicon: `python lang_ident_test.py`


### If you are new to Python then run the following from a terminal to get started:

`git clone https://github.com/praekelt/feersum-lid-shared-task.git`

`cd feersum-lid-shared-task`

`virtualenv -p /usr/local/bin/python3.6 .pyenv`
 (alternatively use python3.7)

`source .pyenv/bin/activate`

`pip install pip-tools`

`pip install appdirs`

`pip install jupyter`

`pip install sklearn`

`pip install numpy`

`pip install scipy`

`pip install requests`





