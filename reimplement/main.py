import re
from nltk.corpus import sentiwordnet as swn
import nltk
from nltk.util import ngrams
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from sklearn import svm

path_to_jar = 'C:\Users\t_quacd\AppData\Local\stanford-parser-full-2015-12-09/stanford-parser.jar'
path_to_models_jar = 'C:\Users\t_quacd\AppData\Local\stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'
parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


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


def lexicon_feature(sentence):  # use wordsentinet to calculate sentimental score. For PMI will do later.
    tokens = nltk.word_tokenize(sentence)
    pos_tokens = nltk.pos_tag(tokens)
    number_pos = 0
    number_neg = 0
    sum_sentimental = 0
    for pos_token in pos_tokens:
        score = 0
        if 'NN' in pos_token[1] and len(swn.senti_synsets(pos_token[0], 'n')) > 0:
            score = (list(swn.senti_synset(pos_token[0], 'n'))[0]).pos_score() - (list(swn.senti_synsets(pos_token[0], 'n'))[0]).neg_score()
        elif 'VB' in pos_token[1] and len(swn.senti_synsets(pos_token[0], 'v')) > 0:
            score = (list(swn.senti_synset(pos_token[0], 'v'))[0]).pos_score() - (list(swn.senti_synsets(pos_token[0], 'v'))[0]).neg_score()
        elif 'JJ' in pos_token[1] and len(swn.senti_synsets(pos_token[0], 'a')) > 0:
            score = (list(swn.senti_synset(pos_token[0], 'a'))[0]).pos_score() - (list(swn.senti_synsets(pos_token[0], 'a'))[0]).neg_score()
        elif 'RB' in pos_token[1] and len(swn.senti_synsets(pos_token[0], 'r')) > 0:
            score = (list(swn.senti_synset(pos_token[0], 'r'))[0]).pos_score() - (list(swn.senti_synsets(pos_token[0], 'r'))[0]).neg_score()
        else:
            continue
        sum_sentimental += score
        if score > 0:
            number_neg += 1
        else:
            number_pos += 1
        if 'not' in tokens:
            sum_sentimental *= -1
    return number_pos,number_neg,sum_sentimental

def surface_feature(window,aspect_term):
    resullt = []
    window = nltk.word_token(window)
    for word in window:
        result.append(word)
        if word != aspect_term:
            result.append(word+aspect_term)
    bigrams = ngrams(window,2)
    result += bigrams
    return result


def parse_feature():


def sentimental_extraction(sentence):  # input is a string


def main():
    with open('restaurants-trial.xml', 'r') as f:
        sentences = re.findall(r'<text>(.*)</', f.read())
    clf = svm.SVC()
    


if __name__ == "__main__":
    main()
