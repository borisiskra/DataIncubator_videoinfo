import pickle
from collections import Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
#nlp = spacy.load("en", disable=['parser', 'ner', 'textcat'])

import en_core_web_sm
nlp = en_core_web_sm.load()
# local imports
from videoinfo import get_transcript
from bert_init import bert_classify_sentences


def bert_video_classifier(videoID):
    # if the video was already clasified,
    # read it from file
    fname = f"data/videos/{videoID}_analysis.pickle"
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)

    except Exception:
        print('Video not analized yet')
        print('trying to get the script...')

    try:
        transcript = get_transcript(videoID)
        # print(transcript)
        common_words = most_common_words(transcript)

        sentence_lst = transcript.split('\n')
        print(f'number sentences: {len(sentence_lst)}')

    except Exception as err:
        print('====>>', err)

    all_topic_predictions = bert_classify_sentences(sentence_lst)

    result = (all_topic_predictions, common_words)
    try:
        with open(fname, 'wb') as f:
            pickle.dump(result, f)

    except Exception as err:
        print('Error: ', err, 'trying to write file')

    return result


def tokenize_lemma(text):
    EXCLUDE = ['PUNCT', 'PRON', 'DET', 'SPACE', 'SCONJ', 'SYM', 'NUM']
    return [w.lemma_.lower() for w in nlp(text) if w.pos_ not in EXCLUDE]


more_stop_words = set(['know', 'include', 'form', 'small', 'large', 'might',
                       'adult', 'where', 'the', 'usually', 'form', 'also',
                       'include', 'know', 'form', 'every', 'enough', 'fully',
                       'show', 'either', 'likelike', 'could', 'would', 'like',
                       'come', 'within', 'oh', 'okay', 'sure', 'fun', 'sure',
                       'that', 'all', 'let', 'nope', 'yup'])

my_stop_words = STOP_WORDS.union(more_stop_words)

stop_words_lemma = set(tokenize_lemma(' '.join(my_stop_words)))


def most_common_words(text):
    Words = Counter()

    # word_lst = re.findall(r"[\w]+", text)
    word_lst = tokenize_lemma(text)

    # print(word_lst)

    for word in word_lst:
        if word == '':
            continue
        if len(word) <= 2:
            continue
        word = word.lower()
        if word in stop_words_lemma:
            continue
        Words[word] += 1

    return Words  # .most_common(5)
