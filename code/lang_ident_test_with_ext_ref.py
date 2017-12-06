import time
from typing import List, Tuple, Dict, Set  # noqa # pylint: disable=unused-import
import requests
import urllib.parse as urllib_parse

import text_classifier


# ==========================
def retrieve_language_ext(service: str, input_text: str) -> List[Tuple[str, float]]:  # (lang_code, score)
    """ Retrieve the language the input text is written in using an external reference API.

    :param service: The http service to use.
    :param input_text: Document/Sentence to analyse.
    :return: A list of probable ISO-639-1 language codes sorted from high to low probability.
    """
    result_list = []  # type: List[Tuple[str, float]]
    # print("retrieve_language_ext: ", service, input_text)

    if input_text is not None and len(input_text) > 0:
        lang_to_iso_map = {"afrikaans": "afr",
                           "english": "eng",
                           "isindebele": "nbl",
                           "isixhosa": "xho",
                           "isizulu": "zul",
                           "sepedi": "nso",
                           "sesotho": "sot",
                           "setswana": "tsn",
                           "siswati": "ssw",
                           "tshivenda": "ven",
                           "xitsonga": "tso"}

        try:
            r = requests.get(service +
                             "?%27" + urllib_parse.quote(input_text) + "%27")  # ,timeout=1.0)

            iso_lang_code = lang_to_iso_map.get(r.content.decode('utf-8').strip().lower())

            if iso_lang_code is None:
                iso_lang_code = "eng"

            result_list.append((iso_lang_code, 1.0))
        except requests.exceptions.RequestException:
            print("retrieve_language_ext exception!")

    return result_list


# ==========================
def add_predicted_lang_labels_ext(service: str,
                                  sent_list: List[Tuple[str, str]]) -> List[Tuple[str, str, List[str]]]:
    """
    Add the predicted language labels to the sentences.

    :param service: The http service to use.
    :param sent_list: The list of sentences labelled with only the truth.
    :return: The list of sentences labelled with the truth and the predicted label.
    """
    sent_list_len = len(sent_list)
    sentence_num = 0

    sent_list_pred_ext = []  # type: List[Tuple[str, str, List[str]]]
    correct_ext = 0

    for sentence, truth in sent_list:
        prediction_ext = retrieve_language_ext(service, sentence)

        if len(prediction_ext) > 0:
            prediction = prediction_ext[0][0]
        else:
            prediction = 'no_response'

        sent_list_pred_ext.append((sentence, truth, [prediction]))

        if prediction == truth:
            correct_ext += 1

        sentence_num += 1

        if truth != prediction:
            print(truth, prediction, sentence, flush=True)
            print()

        if (sentence_num % 100) == 0:
            print(str(round(sentence_num / float(sent_list_len) * 100.0, 2)) + " ", flush=True)
            print("acc =", correct_ext / float(sentence_num))
            print()

    return sent_list_pred_ext


# ==========================
def pred_language_lex(lang_token_dict: Dict[str, Dict[str, int]],
                      input_text: str) -> List[Tuple[str, float]]:
    """
    Use the language lexicons to predict which language the text is written in.

    :param lang_token_dict: The token vs. count lexicon for each language Dict[lang, Dict[word, count]]
    :param input_text: The input text to LID.
    :return: The scored language labels.
    """
    scored_labels = []  # List[Tuple[str, float]]

    input_tokens = input_text.lower().split()

    for lang_code, word_dict in lang_token_dict.items():
        score = 0.0

        for token in input_tokens:
            if word_dict.get(token, 0) > 0:
                score += 1

        scored_labels.append((lang_code, score))

    scored_labels.sort(key=lambda scored_label: scored_label[1], reverse=True)

    return scored_labels


# ==========================
def add_predicted_lang_labels(feat_clsfr: text_classifier.FeatClsfr,
                              lang_token_dict: Dict[str, Dict[str, int]],
                              sent_list: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str, List[str]]],
                                                                         List[Tuple[str, str, List[str]]],
                                                                         List[Tuple[str, str, List[str]]]]:
    """
    Add the predicted language labels to the sentences.

    :param feat_clsfr: The classifier object to use.
    :param lang_token_dict: The token vs. count lexicon for each language Dict[lang, Dict[word, count]]
    :param sent_list: The list of sentences labelled with only the truth.
    :return: The list of sentences labelled with the truth and the predicted label.
    """
    sent_list_len = len(sent_list)
    sentence_num = 0

    sent_list_pred_nb = []  # type: List[Tuple[str, str, List[str]]]
    sent_list_pred_lex = []  # type: List[Tuple[str, str, List[str]]]
    sent_list_pred_cmb = []  # type: List[Tuple[str, str, List[str]]]

    correct_nb = 0
    correct_lex = 0
    correct_cmb = 0

    for sentence, truth in sent_list:
        prediction_nb = text_classifier.retrieve_text_class(feat_clsfr, sentence)
        prediction_lex = pred_language_lex(lang_token_dict, sentence)

        if (prediction_nb[0][0] == 'xho') or (prediction_nb[0][0] == 'zul') or \
                (prediction_nb[0][0] == 'ssw') or (prediction_nb[0][0] == 'nbl'):
            lang_list = ['zul', 'xho', 'nbl', 'ssw']

            scored_lang_list = []  # type: List[Tuple[str, float]]

            for lang, score in prediction_lex:
                if lang in lang_list:
                    scored_lang_list.append((lang, score))

            scored_lang_list.sort(key=lambda scored_label: scored_label[1], reverse=True)

            if scored_lang_list[0][1] > scored_lang_list[1][1]:
                prediction_cmb = scored_lang_list[0][0]
            else:
                prediction_cmb = prediction_nb[0][0]
        elif (prediction_nb[0][0] == 'nso') or (prediction_nb[0][0] == 'sot') or (prediction_nb[0][0] == 'tsn'):
            lang_list = ['nso', 'sot', 'tsn']

            scored_lang_list = []  # type: List[Tuple[str, float]]

            for lang, score in prediction_lex:
                if lang in lang_list:
                    scored_lang_list.append((lang, score))

            scored_lang_list.sort(key=lambda scored_label: scored_label[1], reverse=True)
            if scored_lang_list[0][1] > scored_lang_list[1][1]:
                prediction_cmb = scored_lang_list[0][0]
            else:
                prediction_cmb = prediction_nb[0][0]
        else:
            prediction_cmb = prediction_nb[0][0]

        sent_list_pred_nb.append((sentence, truth, [prediction_nb[0][0]]))
        sent_list_pred_lex.append((sentence, truth, [prediction_lex[0][0]]))
        sent_list_pred_cmb.append((sentence, truth, [prediction_cmb]))

        if prediction_nb[0][0] == truth:
            correct_nb += 1

        if prediction_lex[0][0] == truth:
            correct_lex += 1

        if prediction_cmb == truth:
            correct_cmb += 1

        sentence_num += 1

        if (truth != prediction_nb[0][0]) or (truth != prediction_cmb):  # and (truth != prediction[1][0]):
            print(truth, prediction_nb[:3], sentence, flush=True)
            print(prediction_lex[:3])
            print(prediction_cmb)
            print()

        if (sentence_num % 100) == 0:
            print(str(round(sentence_num / float(sent_list_len) * 100.0, 2)) + " ", flush=True)
            print("acc =", correct_nb / float(sentence_num))
            print("acc_lex =", correct_lex / float(sentence_num))
            print("acc_cmb =", correct_cmb / float(sentence_num))
            print()

    return sent_list_pred_nb, sent_list_pred_lex, sent_list_pred_cmb


# ==========================
def find_unique(lang_token_dict: Dict[str, Dict[str, int]],
                target_lang: str,
                neighbours: Set[str]) -> Set[str]:
    """
    Find the unique set of words in lang that are not in the neighbours.
    :param lang_token_dict: Language lexicon (word vs. count for each language.)
    :param target_lang: The language to analyse.
    :param neighbours: The neighbouring languages to compare against.
    :return: The set of unique words in lang.
    """

    target_lex = lang_token_dict.get(target_lang, None)
    unique_set = set()  # type: Set[str]

    if target_lex is not None:
        for target_token in target_lex:
            neighbour_count = 0

            for neighbour_lang in neighbours:
                neighbour_lex = lang_token_dict.get(neighbour_lang, None)

                if (neighbour_lex is not None) and (target_token in neighbour_lex):
                    neighbour_count += 1

            if neighbour_count == 0:
                unique_set.add(target_token)

    return unique_set


# ==========================
def find_common(lang_token_dict: Dict[str, Dict[str, int]],
                languages: Set[str]) -> Set[str]:
    """
    Find the common set of words in languages.
    :param lang_token_dict: Language lexicon (word vs. count for each language.)
    :param languages: The languages to analyse.
    :return: The set of common words between languages.
    """

    common_tokens = None

    for lang in languages:
        token_dict = lang_token_dict.get(lang)

        if token_dict is not None:
            if common_tokens is None:
                common_tokens = set(token_dict.keys())  # type: Set[str]
            else:
                common_tokens.intersection_update(set(token_dict.keys()))

    return common_tokens


# ==========================
print("Loading the text corpora...")
start_time = time.time()
language_set = {'afr': '../data/afr/improved_afr.txt',
                'eng': '../data/eng/improved_eng.txt',
                'nbl': '../data/nbl/improved_nbl.txt',
                'xho': '../data/xho/improved_xho.txt',
                'zul': '../data/zul/improved_zul.txt',
                'nso': '../data/nso/improved_nso.txt',
                'sot': '../data/sot/improved_sot.txt',
                'tsn': '../data/tsn/improved_tsn.txt',
                'ssw': '../data/ssw/improved_ssw.txt',
                'ven': '../data/ven/improved_ven.txt',
                'tso': '../data/tso/improved_tso.txt'}

training_samples = 3000
testing_samples = 1000

min_requested_sent_length = 15  # value used to truncate the text samples.

sent_list_train, sent_list_test, lang_token_dict = text_classifier.load_sentences_all(language_set,
                                                                                      min_requested_sent_length,
                                                                                      training_samples,
                                                                                      testing_samples)

# text_classifier.save_samples_csv(sent_list_train, 'train_full_3k')
# text_classifier.save_samples_csv(sent_list_test, 'test_full_1k')
# text_classifier.save_samples_csv(sent_list_test, 'test_15_1k')

end_time = time.time()
print('Data loading time = ' + str(end_time - start_time) + 's.')
print()

# ==========================
print("Analysing the lexicons ... ")
start_time = time.time()
full_zul = find_unique(lang_token_dict, 'zul', set())
full_xho = find_unique(lang_token_dict, 'xho', set())
full_ssw = find_unique(lang_token_dict, 'ssw', set())
full_nbl = find_unique(lang_token_dict, 'nbl', set())
unique_zul = find_unique(lang_token_dict, 'zul', {'xho', 'ssw', 'nbl'})
unique_xho = find_unique(lang_token_dict, 'xho', {'zul', 'ssw', 'nbl'})
unique_ssw = find_unique(lang_token_dict, 'ssw', {'xho', 'zul', 'nbl'})
unique_nbl = find_unique(lang_token_dict, 'nbl', {'xho', 'zul', 'ssw'})
common_zul_xho = find_common(lang_token_dict, {'xho', 'zul'})
common_nguni = find_common(lang_token_dict, {'nbl', 'xho', 'zul', 'ssw'})

full_nso = find_unique(lang_token_dict, 'nso', set())
full_sot = find_unique(lang_token_dict, 'sot', set())
full_tsn = find_unique(lang_token_dict, 'tsn', set())
unique_nso = find_unique(lang_token_dict, 'nso', {'sot', 'tsn'})
unique_sot = find_unique(lang_token_dict, 'sot', {'nso', 'tsn'})
unique_tsn = find_unique(lang_token_dict, 'tsn', {'nso', 'sot'})
common_sotho_tswana = find_common(lang_token_dict, {'nso', 'sot', 'tsn'})

print("full_zul({0}):".format(len(full_zul)), full_zul)
print("full_xho({0}):".format(len(full_xho)), full_xho)
print("full_ssw({0}):".format(len(full_ssw)), full_ssw)
print("full_nbl({0}):".format(len(full_nbl)), full_nbl)
print("unique_zul({0}):".format(len(unique_zul)), unique_zul)
print("unique_xho({0}):".format(len(unique_xho)), unique_xho)
print("unique_ssw({0}):".format(len(unique_ssw)), unique_ssw)
print("unique_nbl({0}):".format(len(unique_nbl)), unique_nbl)
print("common_nguni({0}):".format(len(common_nguni)), common_nguni)
print()
print("full_nso({0}):".format(len(full_nso)), full_nso)
print("full_sot({0}):".format(len(full_sot)), full_sot)
print("full_tsn({0}):".format(len(full_tsn)), full_tsn)
print("unique_nso({0}):".format(len(unique_nso)), unique_nso)
print("unique_sot({0}):".format(len(unique_sot)), unique_sot)
print("unique_tsn({0}):".format(len(unique_tsn)), unique_tsn)
print("common_sotho_tswana({0}):".format(len(common_sotho_tswana)), common_sotho_tswana)
print()
end_time = time.time()
print('done. time = ' + str(end_time - start_time) + 's.')

# ==========================
# ===
# Note: Clsfr name convention for baseline long sentence trained classifier is:
# lid_za_clean_240_3k typically implies training samples of avrg 240 chars in length and 3k=3000 samples per language!
# ===
# text_clsfr_name = 'lid_za_clean_240_3k'
text_clsfr_name = 'lid_za_clean_240_4k'

print("Loading the NB LID classifier", text_clsfr_name, "... ")
start_time = time.time()

feat_clsfr = text_classifier.load_text_clsfr(text_clsfr_name)
print("load_result =", (feat_clsfr is not None))
end_time = time.time()
print('done. time = ' + str(end_time - start_time) + 's.')
print()

# ==========================
print("Running LID on test data ... ")
start_time = time.time()
sent_list_pred, sent_list_pred_lex, sent_list_pred_cmb = \
    add_predicted_lang_labels(feat_clsfr, lang_token_dict, sent_list_test)

end_time = time.time()
print('done. Testing time = ' + str(end_time - start_time) + 's.',
      round(len(sent_list_test)/(end_time - start_time), 2), 'operations/s')
print()

# ==========================
print("Running LID_ext on test data ... ")
start_time = time.time()
sent_list_pred_ext = add_predicted_lang_labels_ext("http://your.lid.service/lid.cgi",
                                                   sent_list_test)
end_time = time.time()
print('done. Testing time = ' + str(end_time - start_time) + 's.')
print()


lang_to_family_dict = {'afr': 'germanic',
                       'eng': 'germanic',
                       'zul': 'nguni',
                       'xho': 'nguni',
                       'ssw': 'nguni',
                       'nbl': 'nguni',
                       'nso': 'sotho-tswana',
                       'sot': 'sotho-tswana',
                       'tsn': 'sotho-tswana',
                       'tso': 'tswa–ronga',
                       'ven': 'venda'}

proposed_lang_label_list = ['afr', 'eng', 'zul', 'xho', 'ssw', 'nbl', 'nso', 'sot', 'tsn', 'tso', 'ven']

print("Analysing LID test results ...")
start_time = time.time()

lang_result_list = []  # type: List[Tuple[str, str, List[str]]]
lang_result_list_lex = []  # type: List[Tuple[str, str, List[str]]]
lang_result_list_cmb = []  # type: List[Tuple[str, str, List[str]]]
lang_result_list_ext = []  # type: List[Tuple[str, str, List[str]]]

fam_result_list = []  # type: List[Tuple[str, str, List[str]]]
fam_result_list_lex = []  # type: List[Tuple[str, str, List[str]]]
fam_result_list_cmb = []  # type: List[Tuple[str, str, List[str]]]

for sentence, truth, pred_list in sent_list_pred:
    lang_result_list.append((sentence, truth, pred_list))
    fam_result_list.append((sentence, lang_to_family_dict[truth], [lang_to_family_dict[pred_list[0]]]))

for sentence, truth, pred_list in sent_list_pred_lex:
    lang_result_list_lex.append((sentence, truth, pred_list))
    fam_result_list_lex.append((sentence, lang_to_family_dict[truth], [lang_to_family_dict[pred_list[0]]]))

for sentence, truth, pred_list in sent_list_pred_cmb:
    lang_result_list_cmb.append((sentence, truth, pred_list))
    fam_result_list_cmb.append((sentence, lang_to_family_dict[truth], [lang_to_family_dict[pred_list[0]]]))

for sentence, truth, pred_list in sent_list_pred_ext:
    lang_result_list_ext.append((sentence, truth, pred_list))

lang_acc, lang_f1, lang_confusion_dict = text_classifier.analyse_clsfr_results(lang_result_list)
print("lang_acc, lang_f1", lang_acc, lang_f1)
text_classifier.print_confusion_matrix(lang_confusion_dict, proposed_lang_label_list, 'recall')
text_classifier.print_confusion_matrix(lang_confusion_dict, proposed_lang_label_list, 'precision')
text_classifier.print_confusion_matrix(lang_confusion_dict, proposed_lang_label_list, 'fscore')

print()
print()

fam_acc, fam_f1, fam_confusion_dict = text_classifier.analyse_clsfr_results(fam_result_list)
print("fam_acc, fam_f1", fam_acc, fam_f1)
text_classifier.print_confusion_matrix(fam_confusion_dict, ['germanic',
                                                            'nguni',
                                                            'sotho-tswana',
                                                            'tswa–ronga',
                                                            'venda'], 'recall')

text_classifier.print_confusion_matrix(fam_confusion_dict, ['germanic',
                                                            'nguni',
                                                            'sotho-tswana',
                                                            'tswa–ronga',
                                                            'venda'], 'precision')

text_classifier.print_confusion_matrix(fam_confusion_dict, ['germanic',
                                                            'nguni',
                                                            'sotho-tswana',
                                                            'tswa–ronga',
                                                            'venda'], 'fscore')
print()
print()
print()
print()


lang_acc_lex, lang_f1_lex, lang_confusion_dict_lex = text_classifier.analyse_clsfr_results(lang_result_list_lex)
print("lang_acc_lex, lang_f1_lex", lang_acc_lex, lang_f1_lex)
text_classifier.print_confusion_matrix(lang_confusion_dict_lex, proposed_lang_label_list, 'recall')
text_classifier.print_confusion_matrix(lang_confusion_dict_lex, proposed_lang_label_list, 'precision')
text_classifier.print_confusion_matrix(lang_confusion_dict_lex, proposed_lang_label_list, 'fscore')

print()
print()

fam_acc_lex, fam_f1_lex, fam_confusion_dict_lex = text_classifier.analyse_clsfr_results(fam_result_list_lex)
print("fam_acc_lex, fam_f1_lex", fam_acc_lex, fam_f1_lex)
text_classifier.print_confusion_matrix(fam_confusion_dict_lex, ['germanic',
                                                                'nguni',
                                                                'sotho-tswana',
                                                                'tswa–ronga',
                                                                'venda'], 'recall')
text_classifier.print_confusion_matrix(fam_confusion_dict_lex, ['germanic',
                                                                'nguni',
                                                                'sotho-tswana',
                                                                'tswa–ronga',
                                                                'venda'], 'precision')
text_classifier.print_confusion_matrix(fam_confusion_dict_lex, ['germanic',
                                                                'nguni',
                                                                'sotho-tswana',
                                                                'tswa–ronga',
                                                                'venda'], 'fscore')

print()
print()
print()
print()


lang_acc_cmb, lang_f1_cmb, lang_confusion_dict_cmb = text_classifier.analyse_clsfr_results(lang_result_list_cmb)
print("lang_acc_cmb, lang_f1_cmb", lang_acc_cmb, lang_f1_cmb)
text_classifier.print_confusion_matrix(lang_confusion_dict_cmb, proposed_lang_label_list, 'recall')
text_classifier.print_confusion_matrix(lang_confusion_dict_cmb, proposed_lang_label_list, 'precision')
text_classifier.print_confusion_matrix(lang_confusion_dict_cmb, proposed_lang_label_list, 'fscore')

print()
print()

fam_acc_cmb, fam_f1_cmb, fam_confusion_dict_cmb = text_classifier.analyse_clsfr_results(fam_result_list_cmb)
print("fam_acc_cmb, fam_f1_cmb", fam_acc_cmb, fam_f1_cmb)
text_classifier.print_confusion_matrix(fam_confusion_dict_cmb, ['germanic',
                                                                'nguni',
                                                                'sotho-tswana',
                                                                'tswa–ronga',
                                                                'venda'], 'recall')
text_classifier.print_confusion_matrix(fam_confusion_dict_cmb, ['germanic',
                                                                'nguni',
                                                                'sotho-tswana',
                                                                'tswa–ronga',
                                                                'venda'], 'precision')
text_classifier.print_confusion_matrix(fam_confusion_dict_cmb, ['germanic',
                                                                'nguni',
                                                                'sotho-tswana',
                                                                'tswa–ronga',
                                                                'venda'], 'fscore')

print()
print()
print()
print()

lang_acc_ext, lang_f1_ext, lang_confusion_dict_ext = text_classifier.analyse_clsfr_results(lang_result_list_ext)
print("lang_acc_ext, lang_f1_ext", lang_acc_ext, lang_f1_ext)
text_classifier.print_confusion_matrix(lang_confusion_dict_ext, proposed_lang_label_list)
