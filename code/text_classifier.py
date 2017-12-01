"""
Feersum_nlu TextClassifier class.
"""

from typing import Dict, Tuple, List, Set, Callable, Union, Optional  # noqa # pylint: disable=unused-import
from sklearn import naive_bayes
import sklearn
import sklearn.metrics
import numpy as np
import string
import pickle
import random


# ==========================
class FeatClsfr(object):
    """Structure/Collection class for classifier, feature labels and feature extractor."""

    def __init__(self,
                 clsfr: Optional[naive_bayes.MultinomialNB],
                 clsfr_feature_labels: Set[str],
                 feature_extractor_desc: str) -> None:
        self.clsfr = clsfr
        self.clsfr_feature_labels = clsfr_feature_labels
        self.feature_extractor_desc = feature_extractor_desc

    def get_clsfr(self) -> Optional[naive_bayes.MultinomialNB]:
        """Return the classifier."""
        return self.clsfr

    def get_feature_labels(self) -> Set[str]:
        """Return the full set of classifier feature labels."""
        return self.clsfr_feature_labels

    def get_feature_extractor(self) -> str:
        """Return the configured feature extractor description."""
        return self.feature_extractor_desc


# === Text feature helpers - Feature engineering happens here ===
def _extract_char_ngram_features(text: str, number_of_chars: int) -> Set[str]:
    """Extract each n-char-gram as a feature label. Returns a set of features.

    :param text: Document/Sentence to extract features from.
    :param number_of_chars: The size of the n-gram. number_of_chars=1 extracts single char features.
    :return: A set of extracted features.
    """
    ngram_list = []  # type: List[str]

    start_pos = 0  # type: int
    end_pos = len(text) - (number_of_chars - 1)

    if end_pos > 0:
        for i in range(start_pos, end_pos):
            ngram_list.append(text[i:i + number_of_chars])

    return set(ngram_list)


def _extract_char_bigram_features(text: str) -> Set[str]:
    return _extract_char_ngram_features(text, 2)


def _extract_char_trigram_features(text: str) -> Set[str]:
    return _extract_char_ngram_features(text, 3)


def _extract_char_4gram_features(text: str) -> Set[str]:
    return _extract_char_ngram_features(text, 4)


def _extract_char_5gram_features(text: str) -> Set[str]:
    return _extract_char_ngram_features(text, 5)


def _extract_char_6gram_features(text: str) -> Set[str]:
    return _extract_char_ngram_features(text, 6)


# The feature extractor registry
feature_extractor_registry = {'CHAR_BIGRAMS': _extract_char_bigram_features,
                              'CHAR_TRIGRAMS': _extract_char_trigram_features,
                              'CHAR_4GRAMS': _extract_char_4gram_features,
                              'CHAR_5GRAMS': _extract_char_5gram_features,
                              'CHAR_6GRAMS': _extract_char_6gram_features}


def _cnvrt_text_to_features(list_text: List[Tuple[str, str]],
                            text_feature_extractor: Callable) -> List[Tuple[Set[str], str]]:
    """Convert labeled training text strings to list of labeled training feature sets.

    :param list_text: List of labeled text strings.
    :param text_feature_extractor: Text feature extractor to use.
    :return: List of labeled feature strings
    """
    list_features = []

    for (text, class_label) in list_text:
        list_features.append((text_feature_extractor(text), class_label))

    return list_features


# ==========================
def train_text_clsfr(training_samples_text: List[Tuple[str, str]],
                     feature_extractor_desc: str) -> Optional[FeatClsfr]:
    """
    Train the text classification from example text.

    :param training_samples_text: is a list of training tuples (text, class_label)
    :param feature_extractor_desc: The text feature extractor to use for this named classifier.
    :return: TextClassifier.FeatClsfr
    """
    text_feature_extractor = feature_extractor_registry[feature_extractor_desc]

    # === Convert labeled training text to list of labeled training feature sets ===
    print("  _cnvrt_text_to_features...", end='', flush=True)
    training_samples_feature_set = _cnvrt_text_to_features(training_samples_text,
                                                           text_feature_extractor)
    print("done.")
    # === ===

    # === Collect all the feature labels into one set. ===
    print("  Collect all the feature labels into one set...", end='', flush=True)
    clsfr_feature_label_set = set()
    clsfr_class_label_set = set()

    for (feature_set, class_label) in training_samples_feature_set:
        clsfr_class_label_set.add(class_label)
        for feature_label in feature_set:
            clsfr_feature_label_set.add(feature_label)

    clsfr_feature_label_list = list(clsfr_feature_label_set)
    clsfr_feature_label_list.sort()
    print("len(clsfr_feature_label_list):", len(clsfr_feature_label_list), end=' ', flush=True)
    print("done.")
    # === ===

    # === Train ===
    print("Training in batches...", end='', flush=True)
    if len(training_samples_feature_set) > 0:
        clsfr = naive_bayes.MultinomialNB()

        num_samples = len(training_samples_feature_set)
        batch_size = min(1000, num_samples)

        X = np.zeros((batch_size, len(clsfr_feature_label_list)), dtype=np.bool)
        Y = np.zeros((batch_size, 1), dtype=np.object)

        # Iterate over all training samples ...
        for r, labelled_feature_set in enumerate(training_samples_feature_set):
            training_feature_set, training_class_label = labelled_feature_set
            r_batch = r % batch_size

            Y[r_batch, 0] = training_class_label
            for c, feature_label in enumerate(clsfr_feature_label_list):
                X[r_batch, c] = feature_label in training_feature_set

            # Submit a batch of samples.
            if (r_batch == (batch_size - 1)) or (r == (num_samples-1)):
                clsfr.partial_fit(X, Y, classes=list(clsfr_class_label_set))
                progress = r / (num_samples - 1) * 100.0
                print(round(progress, 2), "% ", end="", flush=True)

        print("done.")

        return FeatClsfr(clsfr, clsfr_feature_label_set, feature_extractor_desc)
    else:
        return None


# ==========================
def retrieve_text_class(feat_clsfr: FeatClsfr,
                        input_text: str) -> List[Tuple[str, float]]:
    """Classify the input text.

    :param feat_clsfr: The feature classifier to use.
    :param input_text: Text/phrase to classify.
    :return: A list of probable topics sorted from highest to lowest probability.
    """
    scored_class_list = []

    if feat_clsfr is not None and feat_clsfr.clsfr is not None:
        text_feature_extractor = feature_extractor_registry[feat_clsfr.feature_extractor_desc]

        clsfr_feature_label_list = list(feat_clsfr.clsfr_feature_labels)
        clsfr_feature_label_list.sort()

        input_feature_set = text_feature_extractor(input_text)

        x = np.zeros((1, len(clsfr_feature_label_list)), dtype=np.bool)
        for c, feature_label in enumerate(clsfr_feature_label_list):
            x[0, c] = feature_label in input_feature_set

        # Run classification and retrieve sorted topic probabilities.
        y_proba_list = feat_clsfr.clsfr.predict_proba(x)

        for i, class_label in enumerate(feat_clsfr.clsfr.classes_):
            scored_class_list.append((class_label, y_proba_list[0, i]))

        scored_class_list.sort(key=lambda topic: topic[1], reverse=True)

    return scored_class_list


# ==========================
def cleanup_text(input_text: str) -> str:
    text = input_text.lower()
    punc_to_remove = string.punctuation.replace('-', '') + '0123456789'
    text = text.translate(str.maketrans(punc_to_remove, ' ' * len(punc_to_remove)))

    text = text.replace('ã…â¡', 'š')
    text = text.replace('ï¿½', '')
    text = text.replace('ª', '')

    text = " ".join(text.split())
    text = text.strip()

    # All special characters are kept.
    return text


# ==========================
def save_text_clsfr(feat_clsfr: FeatClsfr,
                    name: str) -> bool:
    """Save the text classifier to a pickle."""

    if feat_clsfr is None:
        return False

    filename = name + '.tc_pickle'

    try:
        handle = open(filename, 'wb')
        pickle.dump(feat_clsfr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        return True
    except IOError:
        print("text_clsfr save error!")
        return False


# ==========================
def load_text_clsfr(name: str) -> Optional[FeatClsfr]:
    """Load or reload the text classifier from a pickle."""
    filename = name + '.tc_pickle'

    try:
        handle = open(filename, 'rb')
        feat_clsfr = pickle.load(handle)
        handle.close()
        return feat_clsfr
    except IOError:
        print("text_clsfr load error!")
        return None


# ==========================
def analyse_clsfr_results(result_list: List[Tuple[str, str, List[str]]]) -> \
        Tuple[float, float, Dict[str, Dict[str, List[str]]]]:
    """
    Analyse the classifier results.

    :param result_list: is a list of results tuples (text, true_label and predicted_labels/top-n-labels)
    :return: The classifier accuracy, f1 score and confusion matrix.
    """
    labels_true = []  # type: List[str]
    labels_predicted = []  # type: List[str]
    num_matched = 0

    # Sparse confusion matrix ...
    confusion_dict = {}  # type: Dict[str, Dict[str, List[str]]]

    if len(result_list) > 0:
        count = 0

        for result_sample in result_list:
            text = result_sample[0]
            true_label = result_sample[1]  # the matrix row label
            predicted_result_labels = result_sample[2]

            if len(predicted_result_labels) > 0:
                row_dict = confusion_dict.get(true_label)
                if row_dict is None:
                    row_dict = {}

                predicted_label = predicted_result_labels[0]  # the matrix column label

                cell = row_dict.get(predicted_label)
                if cell is None:
                    cell = [text]
                else:
                    cell.append(text)

                row_dict[predicted_label] = cell
                confusion_dict[true_label] = row_dict

                # === Update the global scoring ===
                # if true_label in predicted_result_labels:
                if true_label == predicted_label:
                    num_matched += 1

                labels_true.append(true_label)
                labels_predicted.append(predicted_label)
                # === ===

            if count % 100 == 0:
                print(".", end="", flush=True)
            count += 1

        accuracy = num_matched / len(result_list)
        f1 = sklearn.metrics.f1_score(labels_true, labels_predicted, average='weighted')
        print(".")

        return accuracy, f1, confusion_dict
    else:
        return 0.0, 0.0, {}


# ==========================
def print_confusion_matrix(confusion_dict: Dict[str, Dict[str, List[str]]],
                           proposed_label_list: List[str] = None,
                           matrix_type: str = None) -> None:
    """
    Print the confusion matric. This method also serves as an example of how to use the sparse confusion matrix.

    :param confusion_dict: The sparse confusion matrix stored as a dict of dicts.
    :param proposed_label_list: The proposed order and subset of class labels to use.
    :param type: The type of matrix ('recall', 'precision', 'fscore'). Default is fscore.
    """
    print()
    print("label_list:", proposed_label_list)
    print()

    # Generate the proposed label list if none provided.
    if proposed_label_list is None:
        row_label_set = set()  # type: Set[str]
        column_label_set = set()  # type: Set[str]

        for row_label, row_dict in confusion_dict.items():
            row_label_set.add(row_label)

            for column_label, _ in row_dict.items():
                column_label_set.add(column_label)

        proposed_label_list = list(row_label_set.union(column_label_set))
        proposed_label_list.sort()

    # Calculate the row and column totals.
    row_total_dict = {}  # type: Dict[str, int]
    column_total_dict = {}  # type: Dict[str, int]

    for row_label, row_dict in confusion_dict.items():
        if row_label in proposed_label_list:
            row_total = 0

            for column_label, cell in row_dict.items():
                if column_label in proposed_label_list:
                    cell_len = len(cell)
                    row_total += cell_len
                    column_total = column_total_dict.get(column_label, 0) + cell_len
                    column_total_dict[column_label] = column_total

            row_total_dict[row_label] = row_total

    # Print the confusion matrix
    print("===")
    print("Confusion Matrix:", matrix_type)
    for row_label in proposed_label_list:
        row_dict = confusion_dict.get(row_label, {})
        row_total = row_total_dict.get(row_label, 0)

        for column_label in proposed_label_list:
            column_total = column_total_dict.get(column_label, 0)

            if (row_total > 0) or (column_total > 0):
                cell = row_dict.get(column_label, [])
                count = len(cell)

                if row_total > 0:
                    recall = count / row_total
                else:
                    recall = 0.0

                if column_total > 0:
                    precision = count / column_total
                else:
                    precision = 0.0

                if (precision + recall) > 0.0:
                    fscore = 2.0 * (precision * recall) / (precision + recall)
                else:
                    fscore = 0.0

                if matrix_type == 'recall':
                    cell_value = recall
                elif matrix_type == 'precision':
                    cell_value = precision
                else:
                    cell_value = fscore

                print('\033[%dm' % int(37.0 - round(cell_value * 7)), end='')
                print(round(cell_value, 3), "\t", end='')
            else:
                print('--- \t', end='')

        print('\033[0m')
    print("===")
    print()


def load_sentences(filename: str, label: str,
                   min_requested_length: int) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Load the sentences/lines of text from the text corpora.

    :param filename: Name of the file to load.
    :param label: The label to assign to each sentence.
    :param min_requested_length: The minimum length of the sentence to return.
    :return: List of labelled sentence strings (text, label) and the language lexicon (word vs. count)
    """
    sent_list = []  # type: List[Tuple[str, str]]
    token_dict = {}  # type: Dict[str, int]

    # Iterate over the lines of the file
    with open(filename, 'rt') as f:
        print("Loading sentences from", filename)
        for line in f:
            if not line.startswith("<fn"):
                text = cleanup_text(line.strip())

                # Hard limit - Only use sentences that are 200 - 300 chars long
                #  All the improved sentences are already this length.
                if 200 < len(text) < 300:
                    tokens = text.split()
                    for token in tokens:
                        token_count = token_dict.get(token, 0)
                        token_dict[token] = token_count + 1

                    text_end_i = min_requested_length
                    # Include all the characters of the last word.
                    while (text_end_i < len(text)) and (text[text_end_i] != ' '):
                        text_end_i += 1

                    sent_list.append((text[:text_end_i], label))
                    # sent_list.append((text, label))

    return sent_list, token_dict


# ==========================
def load_sentences_all(language_set: Dict[str, str],
                       min_requested_length: int,
                       training_samples: int,
                       testing_samples: int) -> Tuple[List[Tuple[str, str]],
                                                      List[Tuple[str, str]],
                                                      Dict[str, Dict[str, int]]]:
    """
    Load, shuffle, split train/test & shuffle again over all languages...

    :param language_set: The dict of lang_code vs. file name.
    :param min_requested_length: The minimum requested sentence length.
    :param training_samples: The number of training samples PER LANGUAGE.
    :param testing_samples: The number of testing samples PER LANGUAGE.
    :return: The shuffled labelled training and testing sets over all languages (text, label) and the language
    lexicon (word vs. count)
    """

    sent_list_train = []  # type: List[Tuple[str, str]]
    sent_list_test = []  # type: List[Tuple[str, str]]
    lang_token_dict = {}  # type: Dict[str, Dict[str, int]]

    for lang, file in language_set.items():
        sent_list, token_dict = load_sentences(file, lang, min_requested_length)
        lang_token_dict[lang] = token_dict
        random.shuffle(sent_list)
        sent_list_train.extend(sent_list[:training_samples])
        sent_list_test.extend(sent_list[training_samples:(training_samples + testing_samples)])
        print("lang, len(sent_list) =", lang, len(sent_list))

    random.shuffle(sent_list_train)
    random.shuffle(sent_list_test)

    return sent_list_train, sent_list_test, lang_token_dict


# ==========================
def save_samples_csv(sample_list: List[Tuple[str, str]],
                     name: str) -> bool:
    """
    Save the samples to a csv file.

    :param sent_list: The list of samples (text, label)
    :param name: The filename to use
    :return:
    """
    filename = name + '.csv'

    try:
        with open(filename, 'wt') as f:
            print('lang_id, text', file=f)

            for text, label in sample_list:
                print(label + ', ' + '"' + text + '"', file=f)

        return True
    except IOError:
        print("save_samples_csv save error!")
        return False


