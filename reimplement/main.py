import re
import nltk
import string
import cPickle
import os
import inspect
import numpy as np
import sys
import time
import networkx as nx
from nltk import ngrams
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from xml.etree import cElementTree as ET
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn import grid_search
from random import randint
from pprint import pprint
from copy import deepcopy

dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

# We keep negation words in stop words
stopwords = [var for var in stopwords.words('english') if var not in ['not', 'isn']]

def Tokenize(text):
    return nltk.word_tokenize(text)

def NRC_lexicon():
    # Return dict: the dictionary of NRC lexicon for example dict['good'] = 'positive'
    with open('NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 'r') as f:
        sentences = f.readlines()
        dict = {}
        for sentence in sentences:
            # Choose only the word with sentimental labeled as 'positive' or 'negative'
            if (('01' in sentence) & (('positive' in sentence) | ('negative' in sentence))):
                sentence = sentence.translate(string.maketrans("\t", " "))
                sentence = sentence.translate(string.maketrans("\n", " "))
                sentence = sentence.split(' ')
                dict[sentence[0]] = sentence[1]
    return dict


def MPQA_lexicon():
    # Return dict: the dictionary of MPQA lexicon for example dict['good'] = 'positive'
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
    # Return dict: the dictionary of BL lexicon for example dict['good'] = 'positive'
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


def PositionOfAspectTerm(sentence, aspect_term, from_char, to_char):
    # Return the position of aspect term in a given sentence with given from/to positions of character
    # For example:
    # Input: sentence "But the staff was so horrible to us", aspect_term is "staff", from_char is 8 and to_char is 13
    # Output: from_index is 2 and the to_index is 2. ( Note that this is 0 indexing).
    list_sentence = sentence.split()
    from_index = 0
    to_index = 0
    position = 0
    for each in list_sentence:
    # We calculate the prefix sum of characters ( position) until it is larger than from_char
        if (position + len(each) + 1) <= int(from_char):
            position += len(each) + 1
        else:
            from_index = list_sentence.index(each)
            break
    position = 0
    # If the aspect term is only 1 word so from_index = to_index, if not calculate as same as for from_index
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


def SurfaceContext(sentence, aspect_term, from_char, to_char, window_size):
    # Extract surface context: a window of window_size words surrounding the term
    # For example:
    # Input: sentence "But the staff was so horrible to us", aspect_term is "staff", from_char is 8 and to_char is 13, window_size is 4
    # Output: ['But', 'the', 'staff', 'was', 'so', 'horrible', 'to']
    list_sentence = sentence.split()
    from_index, to_index = PositionOfAspectTerm(sentence, aspect_term, from_char, to_char)
    if (from_index - window_size > 0):
        window_begin = from_index - window_size
    else:
        window_begin = 0
    if to_index + window_size < (len(list_sentence)):
        window_end = to_index + window_size+1
    else:
        window_end = len(list_sentence)
    # print window_begin,window_end
    window = list_sentence[window_begin:window_end]
    return window


def ParseContext(sentence, aspect_term, from_char, to_char):
    # Extract parse context: the nodes in the parse tree that are connected to the target term by at most 3 edges.
    # For example:
    # Input: sentence "If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here."
    #  aspect term "bottle minimum",from_char 105,to_char 119
    # Output: ['then', 'want', 'pay', 'dollar', 'least', 'to', 'minimum', 'bottle', '100', 'you', 'blasted', 'if']
    from_index, to_index = PositionOfAspectTerm(sentence, aspect_term, from_char, to_char)
    source_fromIndex = str(from_index + 1)
    source_toIndex = str(to_index + 1)
    parse_sent = dep_parser.raw_parse(sentence)
    dep = parse_sent.next()
    G = nx.Graph()
    context = []
    for each in dep.to_dot().split("\n")[4:-1]:  # Use index [4:-1] to get the text only
        # With the example, each in dep.to_dot().split("\n")[4:-1] will return:
        # 0[label = "0 (None)"]
        # 0 -> 28[label = "root"]
        # 1[label = "1 (If)"]
        # 2[label = "2 (you)"]
        # 3[label = "3 (like)"]
        # 3 -> 6[label = "advcl"]
        # 3 -> 2[label = "nsubj"]
        # 3 -> 1[label = "mark"]
        # 4[label = "4 (your)"]
        # 5[label = "5 (music)"]
        # 5 -> 4[label = "nmod:poss"]
        # 6[label = "6 (blasted)"]
        # 6 -> 7[label = "cc"]
        # 6 -> 13[label = "cc"]
        # 6 -> 12[label = "conj"]
        # 6 -> 16[label = "conj"]
        # 6 -> 5[label = "nsubj"]
        # 7[label = "7 (and)"]
        # 8[label = "8 (the)"]
        # 9[label = "9 (system)"]
        # 10[label = "10 (isnt)"]
        # 10 -> 8[label = "det"]
        # 10 -> 9[label = "compound"]
        # 11[label = "11 (that)"]
        # 12[label = "12 (great)"]
        # 12 -> 10[label = "dep"]
        # 12 -> 11[label = "advmod"]
        # 13[label = "13 (and)"]
        # 14[label = "14 (if)"]
        # 15[label = "15 (you)"]
        # 16[label = "16 (want)"]
        # 16 -> 18[label = "xcomp"]
        # 16 -> 15[label = "nsubj"]
        # 16 -> 14[label = "mark"]
        # 17[label = "17 (to)"]
        # 18[label = "18 (pay)"]
        # 18 -> 24[label = "dobj"]
        # 18 -> 25[label = "advmod"]
        # 18 -> 17[label = "mark"]
        # 19[label = "19 (at)"]
        # 20[label = "20 (least)"]
        # 20 -> 19[label = "case"]
        # 21[label = "21 (100)"]
        # 21 -> 20[label = "nmod:npmod"]
        # 22[label = "22 (dollar)"]
        # 22 -> 21[label = "dep"]
        # 23[label = "23 (bottle)"]
        # 24[label = "24 (minimum)"]
        # 24 -> 22[label = "dep"]
        # 24 -> 23[label = "compound"]
        # 25[label = "25 (then)"]
        # 26[label = "26 (you)"]
        # 27[label = "27 ('ll)"]
        # 28[label = "28 (love)"]
        # 28 -> 27[label = "aux"]
        # 28 -> 26[label = "nsubj"]
        # 28 -> 29[label = "dobj"]
        # 28 -> 30[label = "advmod"]
        # 28 -> 3[label = "advcl"]
        # 29[label = "29 (it)"]
        # 30[label = "30 (here)"]
        each = str(each) # Convert the input from unicode to text
        #print each
        # Remove all the punctuations except ">","=" to identify the relationship in the graph
        each = each.translate(string.maketrans("", ""), string.punctuation.replace(">", "").replace("=", ""))
        #print each
        if ">" in each: # If there is ">" the text is about the relationship( edge), so we add edge
            relationship = each.split("=")[-1]
            each = each.split()
            G.add_edge(each[0], each[2], label=relationship)
        else: # if not so we add node
            each = each.split()
            G.add_node(each[0], name=each[-1])
    name_attribute = nx.get_node_attributes(G, 'name')
    try:
        parse_context_fromIndex = nx.single_source_shortest_path_length(G, source_fromIndex, cutoff=3) # We return nodes connected to source by at most 3 edges ( cutoff)
        for each in parse_context_fromIndex.keys():
            condition = int(each)
            if (condition > 0 and (condition <= (from_index + 1) or condition >= (to_index + 1))): # To make sure the nodes is not in the aspect term itself
                context.append(name_attribute[each])
        if to_index != from_index:
            parse_context_toIndex = nx.single_source_shortest_path_length(G, source_toIndex, cutoff=3)
            for each in parse_context_toIndex.keys():
                condition = int(each)
                if (condition > 0 and (condition < (from_index + 1) or condition > (to_index + 1))): # To make sure the nodes is not in the aspect term itself
                    context.append(name_attribute[each])
    except:
        print "The sentence cannot be processed: ", sentence
        pass
    return list(set(context))


def LexiconFeatures(sentence):
    # Extract information: number of positive/negative tokens, the sum of the tokens' sentiment scores, and for 3 dictionaries
    # For example:
    # Input: sentence "If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here."
    # Output: [4, 2, 2, 3, 1, 2, 3, 1, 2]
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
    # From SurfaceContext, extract unigrams, bigrams, context-target bigrams ( form from surface context and aspect term
    # Input: sentence "If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here."
    # aspect term "bottle minimum",from_char 105,to_char 119, window_size 10
    # Output: list
    # ['and', 'if', 'you', 'want', 'to', 'pay', 'at', 'least', '100', 'dollar', 'bottle', 'minimum', 'then', "you'll", 'love', 'it', 'here.', 'and_bottle minimum_ct',
    # 'if_bottle minimum_ct', 'you_bottle minimum_ct', 'want_bottle minimum_ct', 'to_bottle minimum_ct', 'pay_bottle minimum_ct', 'at_bottle minimum_ct',
    # 'least_bottle minimum_ct', '100_bottle minimum_ct', 'dollar_bottle minimum_ct', 'bottle minimum_then_ct', "bottle minimum_you'll_ct", 'bottle minimum_love_ct',
    # 'bottle minimum_it_ct', 'bottle minimum_here._ct', 'and_if', 'if_you', 'you_want', 'want_to', 'to_pay', 'pay_at', 'at_least', 'least_100', '100_dollar',
    # 'dollar_bottle', 'bottle_minimum', 'minimum_then', "then_you'll", "you'll_love", 'love_it', 'it_here.', 'bottle minimum_at']

    window = SurfaceContext(sentence, aspect_term, from_char, to_char, window_size)
    sentence_window = ' '.join(window)
    slice = sentence_window.partition(aspect_term)
    result = []
    # Adding unigrams to output
    for i in range(0, len(window)):
        result.append(window[i])
    # Adding bigram context target to output
    for i in range(0, len(window)):
        # Determine whether the word is before or after the aspect term to form the correct bigrams context target (ct)
        if window[i] in slice[0]:
            result.append(window[i] + "_" + aspect_term + "_ct")
        elif window[i] in slice[2]:
            result.append(aspect_term + "_" + window[i] + "_ct")
    # Adding bigrams to output
    bigrams = ngrams(window, 2)
    for bigram in bigrams:
        result.append(bigram[0] + "_" + bigram[1])
    result.append(aspect_term + "_at")
    # Adding aspect term to output(at)
    return result  # Return a list of surface features


def ParseFeatures(sentence, aspect_term, from_char, to_char):
    # From Parse Context, extract word and POS-ngrams in the parse context, context-target bigrams, all paths that start or end with the root of the target terms.
    # For example:
    # Input: sentence "If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here."
    # aspect term "bottle minimum",from_char 105,to_char 119
    # Output: ['then_RB', 'want_VBP', 'pay_VB', 'dollar_NN', 'least_JJS', 'to_TO', 'minimum_JJ', 'bottle_NN', '100_CD', 'you_PRP', 'blasted_VBN', 'if_IN',
    # 'bottle minimum_then_ct', 'want_bottle minimum_ct', 'pay_bottle minimum_ct', 'dollar_bottle minimum_ct', 'least_bottle minimum_ct', 'to_bottle minimum_ct',
    # '100_bottle minimum_ct', 'you_bottle minimum_ct', 'blasted_bottle minimum_ct', 'if_bottle minimum_ct']


    parseContext = ParseContext(sentence, aspect_term, from_char, to_char)
    result = []
    dict_postag = {} # dict_postag = {'word1':'POSTAG1','word2':'POSTAG2'}
    postag = nltk.pos_tag(nltk.word_tokenize(sentence))
    # Create a dictionary with words as keys and POS_TAG as values
    for each in postag:
        dict_postag[each[0]] = each[1]
    for each in parseContext:
        result.append(each + '_' + dict_postag[each])  # Adding Word_POSTAG to the output
    # Adding the context target to the output
    slice = sentence.partition(aspect_term)
    for word in parseContext:
        # Determine whether the word is before or after the aspect term to form the correct bigrams context target
        if word in slice[0]:
            result.append(word + "_" + aspect_term + "_ct")
        elif word in slice[2]:
            result.append(aspect_term + "_" + word + "_ct")
    return result


def SentenceTransform(sentence, aspect_term, from_char, to_char, window_size,features):
    # Form the new sentence after adding all needed features. The para features = 2 is not including parse features, 3 is including parse features
    # For example:
    # Input: sentence "If you like your music blasted and the system isnt that great
    # and if you want to pay at least 100 dollar bottle minimum then you'll love it here.",
    #  aspect term  "bottle minimum",from_char 105,to_char 119, window_size 10
    # Output:
    # then_RB want_VBP pay_VB dollar_NN least_JJS to_TO minimum_JJ bottle_NN 100_CD you_PRP blasted_VBN
    # if_IN bottle minimum_then_ct want_bottle minimum_ct pay_bottle minimum_ct dollar_bottle minimum_ct
    # least_bottle minimum_ct to_bottle minimum_ct 100_bottle minimum_ct you_bottle minimum_ct
    # blasted_bottle minimum_ct if_bottle minimum_ct and if you want to pay at least 100 dollar
    # bottle minimum then you'll love it here. and_bottle minimum_ct if_bottle minimum_ct you_bottle minimum_ct
    #  want_bottle minimum_ct to_bottle minimum_ct pay_bottle minimum_ct at_bottle minimum_ct least_bottle minimum_ct
    # 100_bottle minimum_ct dollar_bottle minimum_ct bottle minimum_then_ct bottle minimum_you'll_ct
    # bottle minimum_love_ct bottle minimum_it_ct bottle minimum_here._ct and_if if_you you_want want_to to_pay pay_at
    # at_least least_100 100_dollar dollar_bottle bottle_minimum minimum_then
    # then_you'll you'll_love love_it it_here. bottle minimum_at
    surface_feats = SurfaceFeatures(sentence, aspect_term, from_char, to_char, window_size)
    if features == 3:
        parse_feats = ParseFeatures(sentence, aspect_term, from_char, to_char)
        return ' '.join(parse_feats + surface_feats)
    else:
        return ' '.join(surface_feats)

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

def PreprocessData(DataDirectory="",PickleDirectory = "",filePickle ="",filename = "",mode_pickle = 0,features = 2):
    if mode_pickle != 2:
        fail_sentence = 0
        polarity = []
        text = []
        # tree = ET.parse('C:/Users/admin/reimplement/reimplement/Data/Restaurants_Train.xml')
        tree = ET.parse(DataDirectory)
        root = tree.getroot()
        for sentence in root.findall('sentence'):
            try:
                if sentence.find('aspectTerms') is None:
                    continue
                content = sentence.find('text').text.translate(string.maketrans("", ""), string.punctuation)
                for aspectTerms in sentence.iter('aspectTerms'):
                    for aspectTerm in aspectTerms.iter('aspectTerm'):
                        text.append(SentenceTransform(content, aspectTerm.get('term').translate(string.maketrans("", ""),
                                                                                                string.punctuation),
                                                      aspectTerm.get("from"), aspectTerm.get("to"), 10,features))
                        polarity.append(aspectTerm.get("polarity"))
            except:
                fail_sentence += 1
                pass
        print "The number of sentences which cannot be processed: ",fail_sentence
        LexFeats = [LexiconFeatures(sent) for sent in text]
        LexFeats = np.array(LexFeats)
        LexFeats = csr_matrix(LexFeats)
        cv = CountVectorizer(tokenizer=Tokenize, dtype=np.float64, binary=False, stop_words=stopwords)
        Y = GetYFromStringLabels(polarity)
        X = cv.fit_transform(text)
        X = Normalizer().fit_transform(X)
        print 'shape of X matrix before adding lex feats', X.shape
        X = hstack([X, LexFeats])
        print 'shape of X matrix after adding lex feats', X.shape
        Vocab = cv.get_feature_names() +['HLPos', 'HLNeg', 'HLSum', 'NrcPos', 'NrcNeg', 'NrcSum', 'SubjPos',
                                                   'SubjNeg', 'SubjSum']
        if mode_pickle != 0:
            data = []
            data.append(X)
            data.append(Y)
            data.append(Vocab)
            namePickle = filename.split('.')[0] + "_using_" + str(features) + "_features"
            cPickle.dump(data, open(PickleDirectory+"/"+namePickle+".txt", 'wb'))
    else:
        loaded = cPickle.load(open(PickleDirectory+"/"+filePickle, 'rb'))
        X = loaded[0]
        Y = loaded[1]
        Vocab = loaded[2]
    return X, Y, Vocab


def GetTopN(W, Vocab, N=20):
    FeatsAndVocab = zip(W.tolist(), Vocab)
    FeatsAndVocab.sort()
    FeatsAndVocab.reverse()
    return FeatsAndVocab[:N]


def AnalyseClassifierFeats(Classifier, Vocab, TopN=20):
    W = deepcopy(Classifier.coef_)
    NegW = W[0, :]
    NeuW = W[1, :]
    PosW = W[2, :]

    TopNeg = GetTopN(NegW, Vocab, TopN)
    TopNeu = GetTopN(NeuW, Vocab, TopN)
    TopPos = GetTopN(PosW, Vocab, TopN)
    # TopConf =  GetTopN(ConfW, Vocab, TopN)
    # return TopNeg, TopNeu, TopPos, TopConf
    return TopNeg, TopNeu,TopPos

def main(temp,DataDirectory="",PickleDirectory="",OutputDirectory=""):
    # Test for 1 sample sentence
    # sentence = "If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here."
    # sentence = sentence.translate(string.maketrans("", ""), string.punctuation)
    # aspect_term = "bottle minimum"
    # from_char = 105
    # to_char = 119
    # print (SentenceTransform(sentence, aspect_term, from_char, to_char,10))
    # # # Test for 1 sample sentence
    # cv = CountVectorizer(tokenizer = Tokenize,dtype=np.float64, binary=False)
    # sentence = ["Pair you food with the excellent beers on tap or their well priced wine list."]
    # sentence[0] = sentence[0].translate(string.maketrans("", ""), string.punctuation)
    # aspect_term = "food"
    # from_char = 9
    # to_char = 13
    # sentence[0] = SentenceTransform(sentence[0], aspect_term, from_char, to_char, 10)
    # X = cv.fit_transform(sentence)
    # X = Normalizer().fit_transform(X)
    # print X.toarray()
    # print cv.get_feature_names()
    # print X.shape
    # print ParseContext("If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here.","bottle minimum",105,119)
    # print SurfaceFeatures(
    #     "If you like your music blasted and the system isnt that great and if you want to pay at least 100 dollar bottle minimum then you'll love it here.",
    #     "bottle minimum", 105, 119,10)
    # print SentenceTransform(
    #     " I would  rather spend my money on a computer that costs more then a Toshiba that  isn't good at all.",
    #     "costs", 52, 57,10,3)
    DataFiles = os.listdir(DataDirectory)
    PickleFiles = os.listdir(PickleDirectory)
    filePickle =""
    print "Using types of features: "
    print "2. Lexicon + Surface Feats"
    print "3. Lexicon + Surface + Parse Feats"
    features =  input("Your choice? ")
    print "Found ",len(DataFiles)," files. Choose one to perform ML"
    i = 1
    for each in DataFiles:
        print i,".",each
        i = i + 1
    choice = input("Your choice? ")
    filename = DataFiles[choice-1]
    print "Found ", len(PickleFiles), " pickled files:"
    i = 1
    for each in PickleFiles:
        print i, ".", each
        i = i + 1
    print
    print "Choose the action:"
    print "1. Not using pickle files"
    print "2. Using pickle files"
    choicePickle = input("Your choice? ")
    if choicePickle==1:
        print"Do you want to pickle the processing of this one? 0. No; 1. Yes"
        select = input("Your choice? ")
    else:
        pick = input("Which files? ")
        filePickle = PickleFiles[pick-1]
        select = 2
    t0 = time.time()
    X, Y, Vocab = PreprocessData(DataDirectory + "/" + DataFiles[choice - 1], PickleDirectory,filePickle , filename, select,features)

    # To write results to the output files
    current_Time = str(time.strftime("%x")).translate(string.maketrans("/","_"))
    orig_stdout = sys.stdout
    f = file(OutputDirectory+ "/" + filename + "_using_"+str(features)+" features_pickle_"+str(choicePickle)+"_"+current_Time+".txt",'w')
    sys.stdout = f
    TopN = 20 # return top 20 the most impactful features
    print "preprocess time:", round(time.time() - t0, 3), "s"
    for i in xrange(5):
        print 'run ', i + 1
        print
        Params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=randint(0, 100))
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        clf = grid_search.GridSearchCV(LinearSVC(class_weight='balanced'), Params, cv=3)
        # clf = LinearSVC(C=1)
        t0 = time.time()
        clf.fit(X_train, Y_train)
        print "training time:", round(time.time() - t0, 3), "s"
        print 'best estimator after 5 fold CV: ', clf.best_estimator_
        # predict
        t0 = time.time()
        pred = clf.predict(X_test)
        print "predicting time:", round(time.time() - t0, 3), "s"  # accuracy
        print "accuracy: ", accuracy_score(Y_test, pred)
        # print the most impact features
        TopNeg, TopNeu, TopPos = AnalyseClassifierFeats(clf.best_estimator_, Vocab, TopN)
        print '*' * 80
        print 'top {} pos feats: '.format(TopN);
        pprint(TopPos);
        print '*' * 80
        print 'top {} neg feats: '.format(TopN);
        pprint(TopNeg);
        print '*' * 80
        print 'top {} neu feats: '.format(TopN);
        pprint(TopNeu);
        print '*' * 80
    sys.stdout = orig_stdout
    f.close()

if __name__ == "__main__":
    # freeze_support()
    print inspect.getcallargs(main, *sys.argv)
    # DataDirectory = C:/Users/admin/reimplement/reimplement/Data
    # PickleDirectory = C:/Users/admin/reimplement/reimplement/Pickle
    # OutputDirectory = C:/Users/admin/reimplement/reimplement/Output
    main(*sys.argv)
    # test = 0
    # main(test,DataDirectory="C:/Users/admin/reimplement/reimplement/Data",PickleDirectory= "C:/Users/admin/reimplement/reimplement/Pickle",OutputDirectory = "C:/Users/admin/reimplement/reimplement/Output")