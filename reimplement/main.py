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
        dict = []
        for sentence in sentences:
            if (('1' in sentence) & (('positive' in sentence) | ('negative' in sentence))):
                sentence = sentence.translate(string.maketrans("\t", " "))
                sentence = sentence.translate(string.maketrans("\n", " "))
                sentence = sentence.split(' ')
                dict.append((sentence[0], sentence[1]))
    dict = set(dict)
    return dict

def MPQA_lexicon():
    with open('subjclueslen1-HLTEMNLP05.txt', 'r') as f:
        sentences = f.readlines()
        dict = []
        for sentence in sentences:
            if (('1' in sentence) & (('positive' in sentence) | ('negative' in sentence))):
                sentence = sentence.translate(string.maketrans("=", " "))
                sentence = sentence.translate(string.maketrans("\n", " "))
                sentence = sentence.split(' ')
                if ((sentence[-2] == ('positive')) |( sentence[-2] == 'negative')): #ignore neutral word for now
                    dict.append((sentence[5], sentence[-2]))
    dict = set(dict)
    return dict

def BL_lexicon():
    dict = []
    with open('negative-words.txt', 'r') as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence = sentence.translate(string.maketrans("\n", " "))
            dict.append((sentence.strip(),'negative'))
    with open('positive-words.txt', 'r') as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence = sentence.translate(string.maketrans("\n", " "))
            dict.append((sentence.strip(),'positive'))
    dict = set(dict)
    return dict

def sentimental_lexicon_ood(): #out of domain lexicons
    lexicon = {}
    dict = list(BL_lexicon()) + list(MPQA_lexicon()) + list(NRC_lexicon())
    for entry in dict:
        lexicon[entry[0]] = 0
    for entry in dict:
        if (entry[1]=='positive'):
            lexicon[entry[0]] += 1
        else:
            lexicon[entry[0]] -= 1
    for entry in lexicon.keys():
        if lexicon[entry] > 0:
            lexicon[entry] = 'positive'
        elif lexicon[entry] == 0:
            lexicon[entry] = 'unknown'
        else:
            lexicon[entry] = 'negative'
    return lexicon

# def sentimental_lexicon_id(): #in domain lexicon


def surface_context(sentence, from_index, to_index,window_size):  # from_index is the begin of the aspect term in sentence
    if from_index - window_size > 0:
        window_begin = from_index - window_size
    else:
        window_begin = 0
    if to_index + window_size <= len(sentence.text):
        window_end = to_index + window_size
    else:
        window_end = len(sentence.text)
    window = sentence[window_begin:window_end]
    return window

def parse_context(sentence,from_index,to_index):

    return result

def lexicon_feature(sentence):  # use wordsentinet to calculate sentimental score. For PMI will do later.
    lexicon  = sentimental_lexicon_ood()
    tokens = nltk.word_tokenize(sentence)
    postag_tokens = nltk.pos_tag(tokens)
    number_pos = 0
    number_neg = 0
    sum_sentimental = 0
    for postag_token in postag_tokens:
        score = 0
        if 'NN' in postag_token[1] and len(swn.senti_synsets(pos_token[0], 'n')) > 0:
            score = (list(swn.senti_synset(postag_token[0], 'n'))[0]).pos_score() - (list(swn.senti_synsets(postag_token[0], 'n'))[0]).neg_score()
        elif 'VB' in postag_token[1] and len(swn.senti_synsets(postag_token[0], 'v')) > 0:
            score = (list(swn.senti_synset(postag_token[0], 'v'))[0]).pos_score() - (list(swn.senti_synsets(postag_token[0], 'v'))[0]).neg_score()
        elif 'JJ' in pos_token[1] and len(swn.senti_synsets(pos_token[0], 'a')) > 0:
            score = (list(swn.senti_synset(postag_token[0], 'a'))[0]).pos_score() - (list(swn.senti_synsets(postag_token[0], 'a'))[0]).neg_score()
        elif 'RB' in pos_token[1] and len(swn.senti_synsets(pos_token[0], 'r')) > 0:
            score = (list(swn.senti_synset(postag_token[0], 'r'))[0]).pos_score() - (list(swn.senti_synsets(postag_token[0], 'r'))[0]).neg_score()
        else:
            continue
        sum_sentimental += score
    for token in tokens:
        if lexicon[token] == "negative":
            number_neg += 1
        elif lexicon[token] == "positive":
            number_pos += 1
        else:
            continue
    if 'not' in tokens:
        sum_sentimental *= -1
    return [number_pos,number_neg,sum_sentimental]

def surface_feature(sentence):
    window = surface_context(sentence,from_index,to_index, window_size)
    result = []
    window = nltk.word_tokenize(window)
    for word in window:
        result.append(word)
        if word != aspect_term:
            result.append(word+aspect_term)
    bigrams = ngrams(window,2)
    for bigram in bigrams:
        result.append(bigram)
    return list(set(result))

# def parse_feature(sentence):
#
#
#
def tokenize(sentence):
    return surface_feature(sentence) + lexicon_feature(sentence) + parse_feature(sentence)

def preprocessData():
    with open('restaurants-trial.xml', 'r') as f:
        sentences = re.findall(r'<text>(.*)</', f.read())
    # vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True, ngram_range=(1, 1),
                                 stop_words=stopwords,
                                 max_df=0.8)
    features_train_transformed = vectorizer.fit_transform(features_train).toarray()
    features_test_transformed = vectorizer.transform(features_test).toarray()
    return features_train_transformed, features_test_transformed, labels_train, labels_test

def main():
    features_train, features_test, labels_train, labels_test= preprocessData()
    clf = svm.SVC()
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time() - t0, 3), "s"
    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time() - t0, 3), "s"  # accuracy
    print metrics.accuracy_score(labels_test, pred)


if __name__ == "__main__":
    main()
