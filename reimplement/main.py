import re
import nltk
import string
import numpy as np
import networkx as nx
from nltk import ngrams
from time import time
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from xml.etree import ElementTree as ET
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn import grid_search
from random import randint

# Call Stanford Parser by nltk, set up is follow the link http://stackoverflow.com/questions/13883277/stanford-parser-and-nltk
# Set up JAVAHOME to jre , CLASSPATH environment variable to run
# Example in this machine:
# CLASSPATH = D:\stanford-parser-full-2015-12-09\stanford-parser-full-2015-12-09\stanford-parser.jar;D:\stanford-parser-full-2015-12-09\stanford-parser-full-2015-12-09\stanford-parser-3.6.0-models.jar
# JAVAHOME = C:\Program Files\Java\jre1.8.0_111\bin

dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

# We keep negation words in stop words
stopwords = [var for var in stopwords.words('english') if var not in ['not', 'isn']]

def Tokenize(text):
    return nltk.word_tokenize(text)

def NRC_lexicon():
    with open('NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 'r') as f:
        sentences = f.readlines()
        dict = {}
        for sentence in sentences:
            # Choose only the word with sentimental labeled as 'positive' or 'negative'
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
                # Ignore neutral word for now
                if ((sentence[-2] == ('positive')) | (sentence[-2] == 'negative')):
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


def PositionOfAspectTerm(sentence, aspect_term, from_char,
                         to_char):  # Return the position of aspect term in a given sentence with given from/to positions of character
    list_sentence = sentence.split()
    from_index = 0
    to_index = 0
    position = 0
    # from_index and to_index is the position of the aspect term in the list
    for each in list_sentence:
        if (position + len(each) + 1) <= int(from_char):
            position += len(each) + 1
        else:
            from_index = list_sentence.index(each)
            break
    position = 0
    # if the aspect term is only 1 word so from_index = to_index
    if len(aspect_term.split()) == 1:
        to_index = from_index
    else:
        for each in list_sentence:
            if position + len(each) + 1 < int(to_char):
                position += len(each) + 1
            else:
                to_index = list_sentence.index(each)
                break
    return from_index, to_index


def SurfaceContext(sentence, aspect_term, from_char, to_char,
                   window_size):  # from_char, to_char is the begin/ end of the aspect term in list sentence.split() extracted from data
    list_sentence = sentence.split()
    from_index, to_index = PositionOfAspectTerm(sentence, aspect_term, from_char, to_char)
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
    return window  # Return a list of n words surrounding the aspect term, n is window_size


def ParseContext(sentence, aspect_term, from_char, to_char):
    from_index, to_index = PositionOfAspectTerm(sentence, aspect_term, from_char, to_char)
    source_fromIndex = str(from_index + 1)
    source_toIndex = str(to_index + 1)
    parse_sent = dep_parser.raw_parse(sentence)
    dep = parse_sent.next()
    G = nx.Graph()
    context = []
    for each in dep.to_dot().split("\n")[4:-1]:  # Use index [4:-1] to get the text only
        # Convert the input from unicode to text
        each = str(each)
        #print each
        # Remove all the punctuations except ">","=" to identify the relationship in the graph
        each = each.translate(string.maketrans("", ""), string.punctuation.replace(">", "").replace("=", ""))
        # print each
        # If there is ">" the text is about the relationship( edge)
        if ">" in each:
            relationship = each.split("=")[-1]
            each = each.split()
            G.add_edge(each[0], each[2], label=relationship)
        else:
            each = each.split()
            G.add_node(each[0], name=each[-1])
    name_attribute = nx.get_node_attributes(G, 'name')
    try:
        parse_context_fromIndex = nx.single_source_shortest_path_length(G, source_fromIndex, cutoff=3)
        for each in parse_context_fromIndex.keys():
            condition = int(each)
            if (condition > 0 and (condition <= (from_index + 1) or condition >= (to_index + 1))):
                context.append(name_attribute[each])
        if to_index != from_index:
            parse_context_toIndex = nx.single_source_shortest_path_length(G, source_toIndex, cutoff=3)
            for each in parse_context_toIndex.keys():
                condition = int(each)
                if (condition > 0 and (condition < (from_index + 1) or condition > (to_index + 1))):
                    context.append(name_attribute[each])
    except:
        print "The sentence cannot be processed: ", sentence
        pass
    return list(set(
        context))  # Return the nodes (words) in the parse tree that are connected to the aspect term by at most three edges


def LexiconFeatures(sentence):
    MPQA_dict = MPQA_lexicon()
    BL_dict = BL_lexicon()
    NRC_dict = NRC_lexicon()
    tokens = nltk.word_tokenize(sentence)
    number_pos_MPQA = 0
    number_neg_MPQA = 0
    number_pos_BL = 0
    number_neg_BL = 0
    number_pos_NRC = 0
    number_neg_NRC = 0

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


def SurfaceFeatures(sentence, aspect_term, from_char, to_char, window_size):
    window = SurfaceContext(sentence, aspect_term, from_char, to_char, window_size)
    sentence_window = ' '.join(window)
    slice = sentence_window.partition(aspect_term)
    result = []
    for i in range(0, len(window)):
        result.append(window[i])
    for i in range(0, len(window)):
        # Determine whether the word is before or after the aspect term to form the correct bigrams context target
        if window[i] in slice[0]:
            result.append(window[i] + "_" + aspect_term + "_ct")
        elif window[i] in slice[2]:
            result.append(aspect_term + "_" + window[i] + "_ct")
    bigrams = ngrams(window, 2)
    for bigram in bigrams:
        result.append(bigram[0] + "_" + bigram[1])
    result.append(aspect_term + "_at")
    return result  # Return a list of surface features


def ParseFeatures(sentence, aspect_term, from_char, to_char):
    parseContext = ParseContext(sentence, aspect_term, from_char, to_char)
    # dict_postag = {'word1':'POSTAG1','word2':'POSTAG2'}
    result = []
    dict_postag = {}
    postag = nltk.pos_tag(nltk.word_tokenize(sentence))
    # Create a dictionary with words as keys and POS_TAG as values
    for each in postag:
        dict_postag[each[0]] = each[1]
    for each in parseContext:
        result.append(each + '_' + dict_postag[each])  # Word_POSTAG in the parse context
    slice = sentence.partition(aspect_term)
    for word in parseContext:
        # Determine whether the word is before or after the aspect term to form the correct bigrams context target
        if word in slice[0]:
            result.append(word + "_" + aspect_term + "_ct")
        elif word in slice[2]:
            result.append(aspect_term + "_" + word + "_ct")
    return result


def SentenceTransform(sentence, aspect_term, from_char, to_char, window_size):
    surface_feats = SurfaceFeatures(sentence, aspect_term, from_char, to_char, window_size)
    parse_feats = ParseFeatures(sentence, aspect_term, from_char, to_char)
    return ' '.join(parse_feats + surface_feats)
    # return ' '.join(surface_feats)


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


def PreprocessData():
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
                text.append(SentenceTransform(content, aspectTerm.get('term').translate(string.maketrans("", ""),
                                                                                        string.punctuation),
                                              aspectTerm.get("from"), aspectTerm.get("to"), 10))
                polarity.append(aspectTerm.get("polarity"))
    LexFeats = [LexiconFeatures(sent) for sent in text]
    LexFeats = np.array(LexFeats)
    LexFeats = csr_matrix(LexFeats)
    cv = CountVectorizer(tokenizer=Tokenize, dtype=np.float64, binary=False, stop_words=stopwords)
    Y = GetYFromStringLabels(polarity)
    X = cv.fit_transform(text).toarray()
    X = Normalizer().fit_transform(X)
    print 'shape of X matrix before adding lex feats', X.shape
    X = hstack([X, LexFeats])
    print 'shape of X matrix after adding lex feats', X.shape
    return X, Y


def main():
<<<<<<< HEAD
    # Test for 1 sample sentence
    sentence = "If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here."
    sentence = sentence.translate(string.maketrans("", ""), string.punctuation)
    aspect_term = "bottle minimum"
    from_char = 105
    to_char = 119
    print (SentenceTransform(sentence, aspect_term, from_char, to_char,10))
=======
    # # Test for 1 sample sentence
    cv = CountVectorizer(tokenizer = Tokenize,dtype=np.float64, binary=False)
    sentence = ["Pair you food with the excellent beers on tap or their well priced wine list."]
    sentence[0] = sentence[0].translate(string.maketrans("", ""), string.punctuation)
    aspect_term = "food"
    from_char = 9
    to_char = 13
    sentence[0] = SentenceTransform(sentence[0], aspect_term, from_char, to_char, 10)
    X = cv.fit_transform(sentence)
    X = Normalizer().fit_transform(X)
    print X.toarray()
    print cv.get_feature_names()
    print X.shape

>>>>>>> d4bfaec94c0cbf24554d2c985888db4a0efd6881


    # t0 = time()
    # X, Y = PreprocessData()
    # print "preprocess time:", round(time() - t0, 3), "s"
    # for i in xrange(5):
    #     print 'run ', i + 1
    #     Params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=randint(0, 100))
<<<<<<< HEAD
=======
    #     # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
>>>>>>> d4bfaec94c0cbf24554d2c985888db4a0efd6881
    #     clf = grid_search.GridSearchCV(LinearSVC(class_weight='balanced'), Params, cv=3)
    #     t0 = time()
    #     clf.fit(X_train, y_train)
    #     print "training time:", round(time() - t0, 3), "s"
    #     print 'best estimator after 5 fold CV: ', clf.best_estimator_
    #     # predict
    #     t0 = time()
    #     pred = clf.predict(X_test)
    #     print "predicting time:", round(time() - t0, 3), "s"  # accuracy
    #     print "accuracy: ", accuracy_score(y_test, pred)


if __name__ == "__main__":
    main()
