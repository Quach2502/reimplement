import re
from nltk.corpus import sentiwordnet as swn
import nltk
from nltk import ngrams
from time import time
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.svm import LinearSVC
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from xml.etree import ElementTree as ET
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import numpy as np
from sklearn import grid_search
from random import randint

# path_to_jar = 'C:\Users\t_quacd\AppData\Local\stanford-parser-full-2015-12-09/stanford-parser.jar'
# path_to_models_jar = 'C:\Users\t_quacd\AppData\Local\stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'
# parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
# dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#
stopwords = [var for var in stopwords.words('english') if var not in ['not', 'isn']]


def NRC_lexicon():
    with open('NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 'r') as f:
        sentences = f.readlines()
        dict = {}
        for sentence in sentences:
            if (('1' in sentence) & (('positive' in sentence) | ('negative' in sentence))):
                sentence = sentence.translate(string.maketrans("\t", " "))
                sentence = sentence.translate(string.maketrans("\n", " "))
                sentence = sentence.split(' ')
                dict[sentence[0]] = sentence[1]
    return dict


def MPQA_lexicon():
    with open('subjclueslen1-HLTEMNLP05.txt', 'r') as f:
        sentences = f.readlines()
        dict = {}
        for sentence in sentences:
            if (('1' in sentence) & (('positive' in sentence) | ('negative' in sentence))):
                sentence = sentence.translate(string.maketrans("=", " "))
                sentence = sentence.translate(string.maketrans("\n", " "))
                sentence = sentence.split(' ')
                if ((sentence[-2] == ('positive')) | (sentence[-2] == 'negative')):  # ignore neutral word for now
                    dict[sentence[5]] = sentence[-2]
    return dict


def BL_lexicon():
    dict = {}
    with open('negative-words.txt', 'r') as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence = sentence.translate(string.maketrans("\n", " "))
            dict[sentence.strip()] = 'negative'
    with open('positive-words.txt', 'r') as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence = sentence.translate(string.maketrans("\n", " "))
            dict[sentence.strip()] = 'positive'
    return dict


# def sentimental_lexicon_id(): #in domain lexicon


def surface_context(sentence, aspect_term, from_char, to_char,
                    window_size):  # from_index is the begin of the aspect term in list sentence.strip()
    list_sentence = sentence.split()
    from_index = 0
    to_index = 0
    position = 0
    for each in list_sentence:
        if (position + len(each) + 1) <= int(from_char):
            position += len(each) + 1
        else:
            from_index = list_sentence.index(each)
            break
    position = 0
    if len(aspect_term.split()) == 1:
        to_index = from_index
    else:
        for each in list_sentence:
            if position + len(each) + 1 < int(to_char):
                position += len(each) + 1
            else:
                to_index = list_sentence.index(each)
                break
    if (from_index - window_size > 0):
        window_begin = from_index - window_size
    else:
        window_begin = 0
    if to_index + window_size < (len(list_sentence)):
        window_end = to_index + window_size
    else:
        window_end = len(list_sentence)
    # print window_begin,window_end
    window = list_sentence[window_begin:window_end]
    return window


# def parse_context(sentence, from_index, to_index):
#     return result


def lexicon_feature(sentence):
    MPQA_dict = MPQA_lexicon()
    BL_dict = BL_lexicon()
    NRC_dict = NRC_lexicon()
    tokens = nltk.word_tokenize(sentence)
    number_pos_MPQA = 0
    number_neg_MPQA = 0
    sum_sentimental_MPQA = 0
    number_pos_BL = 0
    number_neg_BL = 0
    sum_sentimental_BL = 0
    number_pos_NRC = 0
    number_neg_NRC = 0
    sum_sentimental_NRC = 0

    for token in tokens:
        if token in MPQA_dict.keys():
            if MPQA_dict[token] == 'positive':
                number_pos_MPQA += 1
            elif MPQA_dict[token] == 'negative':
                number_neg_MPQA += 1
        if token in BL_dict.keys():
            if BL_dict[token] == 'positive':
                number_pos_BL += 1
            elif BL_dict[token] == 'negative':
                number_neg_BL += 1
        if token in NRC_dict.keys():
            if NRC_dict[token] == 'positive':
                number_pos_NRC += 1
            elif NRC_dict[token] == 'negative':
                number_neg_NRC += 1
    sum_sentimental_MPQA = number_pos_MPQA - number_neg_MPQA
    sum_sentimental_BL = number_pos_BL - number_neg_BL
    sum_sentimental_NRC = number_pos_NRC - number_neg_NRC
    result = [number_pos_MPQA, number_neg_MPQA, sum_sentimental_MPQA, number_pos_BL, number_neg_BL, sum_sentimental_BL,
              number_pos_NRC, number_neg_NRC, sum_sentimental_NRC]
    return result


def sentence_transform(sentence, aspect_term, from_char, to_char, window_size):
    window = surface_context(sentence, aspect_term, from_char, to_char, window_size)
    sentence_window = ' '.join(window)
    slice = sentence_window.partition(aspect_term)
    result = []
    for i in range(0, len(window)):
        result.append(window[i])
    for i in range(0, len(window)):
        if window[i] in slice[0]:
            result.append(window[i] + "_" + aspect_term + "_ct")
        elif window[i] in slice[2]:
            result.append(aspect_term + "_" + window[i] + "_ct")
    bigrams = ngrams(window, 2)
    for bigram in bigrams:
        result.append(bigram[0] + "_" + bigram[1])
    result.append(aspect_term + "_at")
    return ' '.join(result)


# def parse_feature(sentence):
def GetYFromStringLabels(Labels):
    Y = []
    for L in Labels:
        if 'positive' == L:
            Y.append(1)
        elif 'negative' == L:
            Y.append(-1)
        elif 'neutral' == L:
            Y.append(0)
        elif 'conflict' == L:
            Y.append(2)
        else:
            Y.append(0)
    return Y


def preprocessData():
    polarity = []
    text = []
    tree = ET.parse('Restaurants_Train.xml')
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        if sentence.find('aspectTerms') is None:
            continue
        content = sentence.find('text').text.translate(string.maketrans("", ""), string.punctuation)
        for aspectTerms in sentence.iter('aspectTerms'):
            for aspectTerm in aspectTerms.iter('aspectTerm'):
                text.append(sentence_transform(content, aspectTerm.get('term').translate(string.maketrans("", ""),
                                                                                         string.punctuation),
                                               aspectTerm.get("from"), aspectTerm.get("to"), 10))
                polarity.append(aspectTerm.get("polarity"))
    LexFeats = [lexicon_feature(Sent) for Sent in text]
    LexFeats = np.array(LexFeats)
    LexFeats = csr_matrix(LexFeats)
    CountVecter = CountVectorizer(dtype=np.float64, binary=False, max_df=0.95, stop_words=stopwords)
    Y = GetYFromStringLabels(polarity)
    X = CountVecter.fit_transform(text)
    X = Normalizer().fit_transform(X)
    X = hstack([X, LexFeats])
    return X, Y
    # vectorizer
    # vectorizer = TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True, ngram_range=(1, 1),
    #                              stop_words=stopwords,
    #                              max_df=0.8)
    # features_train_transformed = vectorizer.fit_transform(features_train).toarray()
    # features_test_transformed = vectorizer.transform(features_test).toarray()
    # return features_train_transformed, features_test_transformed, labels_train, labels_test


def main():
    t0 = time()
    X, Y = preprocessData()
    print "preprocess time:", round(time() - t0, 3), "s"
    for i in xrange(5):
        print 'run ', i + 1
        Params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=randint(0, 100))
        clf = grid_search.GridSearchCV(LinearSVC(class_weight='balanced'), Params, cv=3)
        t0 = time()
        clf.fit(X_train, y_train)
        print "training time:", round(time() - t0, 3), "s"
        print 'best estimator after 5 fold CV: ', clf.best_estimator_
        # predict
        t0 = time()
        pred = clf.predict(X_test)
        print "predicting time:", round(time() - t0, 3), "s"  # accuracy
        print "accuracy: ",accuracy_score(y_test, pred)


if __name__ == "__main__":
    main()
