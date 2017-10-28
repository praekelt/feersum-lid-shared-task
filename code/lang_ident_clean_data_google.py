""" Example of sentence similarity. """

import time
from typing import List, Tuple  # noqa # pylint: disable=unused-import
import random

from feersum_nlu import nlp_engine
from feersum_nlu import nlp_engine_data

# Imports the Google Cloud client library
from google.cloud import translate

# Instantiates a client
translate_client = translate.Client.from_service_account_json('/Users/bduvenhage/Desktop/acc7680c1251.json')
# Dashboard at https://console.cloud.google.com/apis/api/translate.googleapis.com/overview

# ==========================
nlpe = nlp_engine.NLPEngine(load_duckling=False, requested_logging_path=nlp_engine_data.get_path() + "/../logs")
print('[Feersum NLU version = ' + nlp_engine.get_version() + ']')
print('[NLPEngine data path = ' + nlp_engine_data.get_path() + ']')
print()


# ==========================
def load_sentences(filename: str, label: str) -> List[Tuple[str, str]]:
    """
    Load the sentences/lines of text from the text corpora.

    :param filename: Name of the file to load.
    :param label: The label to assign to each sentence.
    :return: List of labelled sentence strings.
    """
    sent_list = []  # type: List[Tuple[str, str]]

    # Iterate over the lines of the file
    with open(filename, 'rt') as f:
        print("Loading sentences from", filename)
        for line in f:
            if not line.startswith("<fn"):
                text = nlpe.cleanup_text(line.strip())

                # if text != '':
                if 200 < len(text) < 300:
                    text_end_i = 100  # len(text)

                    while (text_end_i < len(text)) and (text[text_end_i] != ' '):
                        text_end_i += 1

                    sent_list.append((text[:text_end_i], label))
                    # sent_list.append((text, label))

    return sent_list


# ==========================
def save_sentences(filename: str, label_to_save: str, labelled_sentences: List[Tuple[str, str]]) -> None:
    """
    Save the sentences to a text corpora.

    :param filename: Name of the file to load.
    :param labelled_sentences:
    """
    with open(filename, 'wt') as f:
        print("Saving sentences to", filename)
        for sentence, label in labelled_sentences:
            if label == label_to_save:
                f.write(sentence + '\n')


# ==========================
print("Loading the text corpora...")
start_time = time.time()

labelled_sentences = load_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/afr/improved.txt",
                                    "afr")
labelled_sentences.extend(
    load_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/eng/improved.txt",
                   "eng"))
labelled_sentences.extend(
    load_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/xho/improved.txt",
                   "xho"))
labelled_sentences.extend(
    load_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/zul/improved.txt",
                   "zul"))
labelled_sentences.extend(
    load_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/sot/improved.txt",
                   "sot"))

# random.seed(0)
random.shuffle(labelled_sentences)

end_time = time.time()
print('Data loading time = ' + str(end_time - start_time) + 's.')
print()

cleaned_labelled_sentence = []
gc_2_ctext = {'en': 'eng', 'af': 'afr', 'st': 'sot', 'zu': 'zul', 'xh': 'xho'}

load_result = nlpe.load_text_clsfr("lid_za_ref", "")
print("load_result =", load_result)

print("Processing the text corpora...")
start_time = time.time()

num_correct_google = 0
num_correct_nb = 0
num_diff = 0

with open('google_labelled.txt', 'wt') as f:
    for sentence, lang_code in labelled_sentences:
        start_time = time.time()

        language = None

        try:
            language = translate_client.detect_language([sentence])
        except Exception as e:
            print('Google cloud client exception:', str(e))
            time.sleep(1.0)
            translate_client = translate.Client.from_service_account_json('/Users/bduvenhage/Desktop/99b13881fdaf.json')

        if language is not None:
            pred_lang_code_google = gc_2_ctext.get(language[0]['language'], language[0]['language'])
        else:
            pred_lang_code_google = '???'

        pred = nlpe.retrieve_text_class("lid_za_ref", sentence)
        pred_lang_code_lid_za = pred[0][0]

        cleaned_labelled_sentence.append((sentence, pred_lang_code_google))
        # cleaned_labelled_sentence.append((sentence, pred_lang_code_lid_za))

        if pred_lang_code_lid_za == lang_code:
            num_correct_nb += 1

        if pred_lang_code_google == lang_code:
            num_correct_google += 1

        if pred_lang_code_lid_za != pred_lang_code_google:
            print("DIFF")
            num_diff += 1

        line = 'acc:' + str(num_correct_google / float(len(cleaned_labelled_sentence))) + \
               ' acc_nb:' + str(num_correct_nb / float(len(cleaned_labelled_sentence))) + \
               ' progress:' + str(len(cleaned_labelled_sentence)) + \
               '(' + str(num_diff) + ')' + '/' + str(len(labelled_sentences)) + \
               ' language:' + str(pred_lang_code_google) + ',' + str(pred_lang_code_lid_za) + ',' + str(pred[0][1]) + \
               ' ' + sentence

        print(line)
        f.write(line + '\n')

        duration = time.time() - start_time
        time_to_sleep = len(sentence) / 550.0 - duration

        if time_to_sleep > 0.0:
            time.sleep(time_to_sleep)

print('done. Processing time = ' + str(end_time - start_time) + 's.')
print()

print("Saving the cleaned text corpora...")
start_time = time.time()

save_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/afr/improved_google.txt", "afr",
               cleaned_labelled_sentence)
save_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/eng/improved_google.txt", "eng",
               cleaned_labelled_sentence)
save_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/xho/improved_google.txt", "xho",
               cleaned_labelled_sentence)
save_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/zul/improved_google.txt", "zul",
               cleaned_labelled_sentence)
save_sentences("/Users/bduvenhage/myWork/dev/Praekelt/data/language_ctext/text/sot/improved_google.txt", "sot",
               cleaned_labelled_sentence)
print('done. Saving time = ' + str(end_time - start_time) + 's.')
print()
