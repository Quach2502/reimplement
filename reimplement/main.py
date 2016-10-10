import re
from nltk.corpus import sentiwordnet as swn
import nltk
from nltk import ngrams
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from sklearn import svm
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

path_to_jar = 'C:\Users\t_quacd\AppData\Local\stanford-parser-full-2015-12-09/stanford-parser.jar'
path_to_models_jar = 'C:\Users\t_quacd\AppData\Local\stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'
parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

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


def surface_context(sentence, aspect_term, window_size):  # from_index is the begin of the aspect term in sentence
    aspect_term_split = aspect_term.split()
    list_sentence = sentence.split()
    if len(aspect_term_split) == 1:
        from_index = list_sentence.index(aspect_term_split[0])
        to_index = from_index
    else:
        from_index = list_sentence.index(aspect_term_split[0])
        to_index = list_sentence.index(aspect_term_split[-1])
    # print from_index, to_index
    if from_index - window_size > 0:
        window_begin = from_index - window_size
    else:
        window_begin = 0
    if to_index + window_size < (len(list_sentence)):
        window_end = to_index + window_size
    else:
        window_end = len(list_sentence) - 1
    # print window_begin,window_end
    window = list_sentence[window_begin:(window_end + 1)]
    return window


def parse_context(sentence, from_index, to_index):
    return result


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


def sentence_transform(sentence, aspect_term, window_size):
    window = surface_context(sentence, aspect_term, window_size)
    aspect_term_split = aspect_term.split()
    if len(aspect_term_split) == 1:
        from_index = window.index(aspect_term_split[0])
        to_index = from_index
    else:
        from_index = window.index(aspect_term_split[0])
        to_index = window.index(aspect_term_split[-1])
    result = []
    for i in range(0, len(window)):
        result.append(window[i])
    for i in range(0, len(window)):
        if i < from_index:
            result.append(window[i] + "_" + aspect_term + "_ct")
        elif i > to_index:
            result.append(aspect_term + "_" + window[i] + "_ct")
    bigrams = ngrams(window, 2)
    for bigram in bigrams:
        result.append(bigram[0] +"_"+bigram[1])
    return ' '.join(result)


# def parse_feature(sentence):
#
#
#
def tokenize(sentence):
    return surface_feature(sentence) + lexicon_feature(sentence) + parse_feature(sentence)


def preprocessData():
    with open('restaurants-trial.xml', 'r') as f:
        sentences = re.findall(r'<sentence (.*)</', f.read().translate(string.maketrans("\n", " ")))
    print sentences
    return
    # vectorizer
    # vectorizer = TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True, ngram_range=(1, 1),
    #                              stop_words=stopwords,
    #                              max_df=0.8)
    # features_train_transformed = vectorizer.fit_transform(features_train).toarray()
    # features_test_transformed = vectorizer.transform(features_test).toarray()
    # return features_train_transformed, features_test_transformed, labels_train, labels_test


def main():
    print sentence_transform('I like the good movie very much but the service is terrible', 'movie', 3)
    preprocessData()
    # features_train, features_test, labels_train, labels_test = preprocessData()
    # clf = svm.SVC()
    # t0 = time()
    # clf.fit(features_train, labels_train)
    # print "training time:", round(time() - t0, 3), "s"
    # # predict
    # t0 = time()
    # pred = clf.predict(features_test)
    # print "predicting time:", round(time() - t0, 3), "s"  # accuracy
    # print metrics.accuracy_score(labels_test, pred)


if __name__ == "__main__":
    main()
