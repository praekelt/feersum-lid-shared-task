import time
from typing import List, Tuple, Dict  # noqa # pylint: disable=unused-import

import text_classifier


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

        if (sentence_num % (sent_list_len / 500)) == 0:
            print(str(round(sentence_num / float(sent_list_len) * 100.0, 2)) + " ", flush=True)
            print("acc =", correct_nb / float(sentence_num))
            print("acc_lex =", correct_lex / float(sentence_num))
            print("acc_cmb =", correct_cmb / float(sentence_num))
            print()

    return sent_list_pred_nb, sent_list_pred_cmb


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

min_requested_sent_length = 300  # value used to truncate the text samples.

sent_list_train, sent_list_test, lang_token_dict = text_classifier.load_sentences_all(language_set,
                                                                                      min_requested_sent_length,
                                                                                      training_samples,
                                                                                      testing_samples)

text_classifier.save_samples_csv(sent_list_train, 'train_full_3k')
text_classifier.save_samples_csv(sent_list_test, 'test_full_1k')
# text_classifier.save_samples_csv(sent_list_test, 'test_15_1k')

end_time = time.time()
print('Data loading time = ' + str(end_time - start_time) + 's.')
print()

avrg_sentence_length = 0.0
for text, label in sent_list_train:
    avrg_sentence_length += len(text)
avrg_sentence_length /= len(sent_list_train)
print('avrg_sentence_length =', avrg_sentence_length)
print()


# ==========================
print("Loading the NB LID classifier ... ")
start_time = time.time()
text_clsfr_name = 'lid_za_clean_200_3k'

feat_clsfr = text_classifier.load_text_clsfr(text_clsfr_name)
print("load_result =", (feat_clsfr is not None))
end_time = time.time()
print('done. time = ' + str(end_time - start_time) + 's.')
print()


# ==========================
print("Running LID on test data ... ")
start_time = time.time()
sent_list_pred, sent_list_pred_cmb = add_predicted_lang_labels(feat_clsfr, lang_token_dict, sent_list_test)
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
lang_result_list_cmb = []  # type: List[Tuple[str, str, List[str]]]

fam_result_list = []  # type: List[Tuple[str, str, List[str]]]
fam_result_list_cmb = []  # type: List[Tuple[str, str, List[str]]]

for sentence, truth, pred_list in sent_list_pred:
    lang_result_list.append((sentence, truth, pred_list))
    fam_result_list.append((sentence, lang_to_family_dict[truth], [lang_to_family_dict[pred_list[0]]]))

for sentence, truth, pred_list in sent_list_pred_cmb:
    lang_result_list_cmb.append((sentence, truth, pred_list))
    fam_result_list_cmb.append((sentence, lang_to_family_dict[truth], [lang_to_family_dict[pred_list[0]]]))

lang_acc, lang_f1, lang_confusion_dict = text_classifier.analyse_clsfr_results(lang_result_list)
print("lang_acc, lang_f1", lang_acc, lang_f1)
text_classifier.print_confusion_matrix(lang_confusion_dict, proposed_lang_label_list)

print()
print()

fam_acc, fam_f1, fam_confusion_dict = text_classifier.analyse_clsfr_results(fam_result_list)
print("fam_acc, fam_f1", fam_acc, fam_f1)
text_classifier.print_confusion_matrix(fam_confusion_dict, ['germanic',
                                                            'nguni',
                                                            'sotho-tswana',
                                                            'tswa–ronga',
                                                            'venda'])

print()
print()
print()
print()

lang_acc_cmb, lang_f1_cmb, lang_confusion_dict_cmb = text_classifier.analyse_clsfr_results(lang_result_list_cmb)
print("lang_acc_cmb, lang_f1_cmb", lang_acc_cmb, lang_f1_cmb)
text_classifier.print_confusion_matrix(lang_confusion_dict_cmb, proposed_lang_label_list)

print()
print()

fam_acc_cmb, fam_f1_cmb, fam_confusion_dict_cmb = text_classifier.analyse_clsfr_results(fam_result_list_cmb)
print("fam_acc_cmb, fam_f1_cmb", fam_acc_cmb, fam_f1_cmb)
text_classifier.print_confusion_matrix(fam_confusion_dict_cmb, ['germanic',
                                                                'nguni',
                                                                'sotho-tswana',
                                                                'tswa–ronga',
                                                                'venda'])

print()
print()
print()
print()
