import time
from typing import List, Tuple  # noqa # pylint: disable=unused-import

import text_classifier


# ==========================
def add_predicted_lang_labels(feat_clsfr: text_classifier.FeatClsfr,
                              sent_list: List[Tuple[str, str]]) -> List[Tuple[str, str, List[str]]]:
    """
    Add the predicted language labels to the sentences.

    :param feat_clsfr: The classifier object to use.
    :param sent_list: The list of sentences labelled with only the truth.
    :return: The list of sentences labelled with the truth and the predicted label.
    """
    sent_list_len = len(sent_list)
    sentence_num = 0

    sent_list_pred = []  # type: List[Tuple[str, str, List[str]]]

    for sentence, truth in sent_list:
        prediction = text_classifier.retrieve_text_class(feat_clsfr, sentence)

        sent_list_pred.append((sentence, truth, [prediction[0][0]]))

        if truth != prediction[0][0]:
            print(truth, prediction[:3], sentence)
            print()

        # print some progress info.
        sentence_num += 1
        if (sentence_num % (sent_list_len / 10)) == 0:
            print(".", end="", flush=True)

    return sent_list_pred


# ==========================
print("Loading the text corpora...")
start_time = time.time()
# language_set = {'afr': '../data/afr/original_NCHLT_afr_CLEAN.2.0.txt',
#                 'eng': '../data/eng/original_NCHLT_eng_CLEAN.1.0.0.txt',
#                 'nbl': '../data/nbl/original_NCHLT_nbl_CLEAN.2.0.txt',
#                 'xho': '../data/xho/original_NCHLT_xho_CLEAN.2.0.txt',
#                 'zul': '../data/zul/original_NCHLT_zul_CLEAN.2.0.txt',
#                 'nso': '../data/nso/original_NCHLT_nso_CLEAN.2.0.txt',
#                 'sot': '../data/sot/original_NCHLT_sot_CLEAN.2.0.txt',
#                 'tsn': '../data/tsn/original_NCHLT_tsn_CLEAN.2.0.txt',
#                 'ssw': '../data/ssw/original_NCHLT_ssw_CLEAN.2.0.txt',
#                 'ven': '../data/ven/original_NCHLT_ven_CLEAN.2.0.txt',
#                 'tso': '../data/tso/original_NCHLT_tso_CLEAN.2.0.txt'}
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

training_samples = 4000
testing_samples = 100

min_requested_sent_length = 300  # value used to truncate the text samples.

# text_clsfr_name = "lid_za_clean_100_1k"  # 0.9959
# text_clsfr_name = "lid_za_clean_100_2k"  # 0.9972
# text_clsfr_name = "lid_za_clean_100_3k"  # 0.9971
# text_clsfr_name = "lid_za_clean_200_1k"  # 0.9994
# text_clsfr_name = "lid_za_clean_200_2k"  # 0.9997
# text_clsfr_name = "lid_za_clean_200_3k"  # 0.9996
# text_clsfr_name = "lid_za_clean_240_1k"  # 0.9998
# text_clsfr_name = "lid_za_clean_240_2k"  # 1.0000
# text_clsfr_name = "lid_za_clean_240_3k"  # 0.9999
# text_clsfr_name = "lid_za_clean_240_4k"  # 1.0

# text_clsfr_name = "lid_za_clean_240_4k_4gram"
text_clsfr_name = "lid_za_clean_240_4k_6gram"


sent_list_train, sent_list_test, lang_token_dict = text_classifier.load_sentences_all(language_set,
                                                                                      min_requested_sent_length,
                                                                                      training_samples,
                                                                                      testing_samples)

# text_classifier.save_samples_csv(sent_list_train, 'train_full_3k')
# text_classifier.save_samples_csv(sent_list_test, 'test_full_1k')

end_time = time.time()
print('Data loading time = ' + str(end_time - start_time) + 's.')
print()


# ==========================
print("Training the text classifier", text_clsfr_name, "...")
start_time = time.time()

# feat_clsfr = text_classifier.train_text_clsfr(sent_list_train, 'CHAR_5GRAMS')
feat_clsfr = text_classifier.train_text_clsfr(sent_list_train, 'CHAR_6GRAMS')

end_time = time.time()
save_result = text_classifier.save_text_clsfr(feat_clsfr, text_clsfr_name)
print("save_result =", save_result)
print('done. Training time = ' + str(end_time - start_time) + 's.')
print()

# ==========================
print("Running LID on test data ... ")
start_time = time.time()
sent_list_pred = add_predicted_lang_labels(feat_clsfr, sent_list_test)
end_time = time.time()
print('done. Testing time = ' + str(end_time - start_time) + 's.')
print()

# ==========================
print("Analysing LID test results ...")
start_time = time.time()
result_list = []  # type: List[Tuple[str, str, List[str]]]

for sentence, truth, pred_list in sent_list_pred:
    result_list.append((sentence, truth, pred_list))

acc, f1, confusion_dict = text_classifier.analyse_clsfr_results(result_list)
print("acc, f1", acc, f1)
text_classifier.print_confusion_matrix(confusion_dict, ['afr', 'eng',
                                                        'zul', 'xho', 'ssw', 'nbl',
                                                        'nso', 'sot', 'tsn',
                                                        'tso',
                                                        'ven'])
end_time = time.time()
print('done. Analysis time = ' + str(end_time - start_time) + 's.')
print()
