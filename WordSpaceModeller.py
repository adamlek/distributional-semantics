# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:21:38 2016
@author: Adam Ek

WEIGHTING:
TODO: !!! TEST double-normalization

MAJOR:
add support for LSA/HAL/neural networks

"""
import sklearn.metrics.pairwise as pw
from stemming.porter2 import stem #ENGLISH
import numpy as np

from collections import defaultdict
import re
import random
import math

class DataReader():
    """
    Reads data from .txt files and organizes them into sentences.

    PARAMS:
        Numerize: Not us use (default: False)

    INPUT:
        preprocess_data: List of .txt files
        sentencizer: Line of text
        word_formatter: string

    OUTPUT:
        List of sentences, list of words in vocabulary, dictionary of documents with wordcount in them
    """
    def __init__(self):
        self.vocabulary = []
        self.documents = defaultdict(dict)
        self.current_doc = ""

    #< By default create vectors from data
    def preprocess_data(self, filenames, numerize = False):
        print('Processing data...')
        sentences_collection = []

        for filename in filenames:
            try:
                with open(filename) as datafile:
                    print('Reading file {0}...'.format(filename))
                    filen = re.search('(\/+.+\/)(.*.txt)', filename)
                    self.current_doc = filen.group(2)
                    self.documents[filen.group(2)] = defaultdict(int)

                    for line in datafile:
                        #< separate row into sentences
                        sentences = self.sentencizer(line)
                        for sentence in sentences:
                            if sentence:
                                formatted_sentence = []
                                for word in sentence:

                                    if '-' in word:
                                        words = word.split('-')
                                        for wordn in words:
                                            formatted_sentence.append(self.word_formatter(wordn.lower()))
                                        continue

                                    formatted_sentence.append(self.word_formatter(word.lower()))

                                while '' in formatted_sentence:
                                    formatted_sentence.remove('')

                                sentences_collection.append(formatted_sentence)
                    print('Success!\n')

            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                continue

        return sentences_collection, self.vocabulary, self.documents

    #< Create sentences from a line in a document
    def sentencizer(self, line):
        start_sent = 0
        sentences = []

        for i, symbol in enumerate(line):

            if i+1 == len(line):
                sentences.append(line[start_sent:].split())

            elif symbol.isupper():
                if line[i+1] != '.' and line[i+1].islower():
                    start_sent = i

            if i+2 >= len(line):
                sentences.append(line[start_sent:-1].split())
                break

            elif symbol == '.' or symbol == '?' or symbol == '!':
                if line[i+2].isupper():
                    sentences.append(line[start_sent:i].split())

        return sentences

    #< Format words
    #< word: self-preservation => self preservation
    #< nums: 5-6 => 56 => NUM, 3.1223 => 31223 => NUM
    def word_formatter(self, word):
        #< remove special things inside words
        word = re.sub('[^a-zåäö0-9%]', '', word)

        #< stem and replace word
        word = stem(word)

        #< dont add null words
        if word == '':
            return ''

        #< FINE TUNE DATA
        #< change numbers to NUM
        if word.isdigit():
            word = 'NUM'
        #< 12% etc => PERC
        elif '%' in word:
            word = 'PERC'

        if word not in self.vocabulary:
            self.vocabulary.append(word)
            self.documents[self.current_doc][word] = 1
        else:
            self.documents[self.current_doc][word] += 1

        return word

#TODO: Add support to make classical word space models
#- word-vector dimensionality, dimensionality reduction: SVD [after/in Contexter]
#TODO: Add training stuff, such that vectors can be reused easier
class RandomVectorizer():
    """
    Creates word and random vectors from a vocabulary(list of words)

    PARAMS:
        dimensions: dimensionality of the random/word vector (default: 2048)
        random_elements: how many random indices to insert +1's and -1's into in the random vector (default: 6)

    INPUT:
        vocabulary_vectorizer: List of words
        random_vector: None

    OUTPUT:
        Dictionary of words in vocabulary with a word_vector and a random_vector
    """
    def __init__(self, dimensions = 2048, random_elements = 6):
        self.dimensions = dimensions
        self.random_elements = random_elements
        self.vocabulary = defaultdict(dict)

    def vocabulary_vectorizer(self, word_list):
        for word in word_list:
            self.vocabulary[word]['word_vector'] = np.zeros(self.dimensions)
            self.vocabulary[word]['random_vector'] = self.random_vector()

        return self.vocabulary

    #< Generate a random vector
    def random_vector(self):
        arr = np.zeros(self.dimensions)

        #< distribute (+1)'s and (-1)'s at random indices
        for i in range(0, self.random_elements):
            if i%2 == 0:
                arr[random.randint(0, self.dimensions-1)] = 1
            else:
                arr[random.randint(0, self.dimensions-1)] = -1

        return arr

#TODO test updates
class Weighter():
    """
    Weights vector based on tf-idf
    https://en.wikipedia.org/wiki/Tf%E2%80%93idf

    Weighter.weight_setup for weight_setup

    SCHEMES:

    scheme 0: standard
        freq/doc_freq * log(N/n)

    scheme 1: double normalization
        0.5 + (0.5 * ((freq/doc_freq)/(max_freq*(max_freq/doc_freq)))) * log(N/n)

    scheme 2: log normalization
        1 + log10(freq/doc_freq) * log(N/n)

    PARAMS:
        scheme: select a weighting scheme to use (default: 0)
        smooth_idf: smooth the idf weight log(1+(N/n))(default: False)
        doidf: compute idf (default: True)

    INPUT:
        INIT: scheme: weighting scheme, arbitrary atms
            document_dict: dictionary of documents and their word count

        METHODS:
        weight: word, random vector
        weight_list: list of strings
        tf: string
        idf: string

    OUTPUT:
        tf-idf weighted vector
    """
    def __init__(self, document_dict, scheme = 0, smooth_idf = False, doidf = True):
        self.document_dict = document_dict
        self.documents_n = len(document_dict)
        self.word_weights = defaultdict(int)

        self.smooth_idf = smooth_idf
        self.scheme = scheme
        self.doidf = doidf

        if self.doidf:
            self.weight_setup = 'tf-idf'
        else:
            self.weight_setup = 'tf'

        if scheme == 1:
            self.weight_setup += ' double norm'
        elif scheme == 2:
            self.weight_setup += ' log norm'
        else:
            pass


    def weight(self, word, vector):
        weight = self.tf(word)
        if self.documents_n > 1:
            if self.doidf:
                weight *= self.idf(word)

        #< save weights for lookup
        self.word_weights[word] = weight
        #< return vector * tf-idf, save tfidf value for future info???
        return vector * weight

    def weight_list(self, wordlist):
        weight_dict = defaultdict(float)
        for word in wordlist:
            weight = self.tf(word)
            if self.documents_n > 1:
                if self.doidf:
                    weight *= self.idf(word)

            self.word_weights[word] = weight

        return self.word_weights

    def tf(self, word):
        tf = []
        for doc in self.document_dict:
            if word in self.document_dict[doc]:
                #< log10 normalization
                if self.scheme == 1:
                    tf.append(1 + math.log10(self.document_dict[doc][word]/sum(self.document_dict[doc].values())))
                #< Double normalization
                elif self.scheme == 2:
                    max_val = max(self.document_dict[doc].values())
                    w1 = self.document_dict[doc][word]/sum(self.document_dict[doc].values())
                    w2 = max_val*(max_val/sum(self.document_dict[doc].values()))

                    tf.append(0.5 + (0.5 * (w1/w2)))
                #< raw term frequency, (freq/doc_freq)
                else:
                    tf.append(self.document_dict[doc][word]/sum(self.document_dict[doc].values()))

        return sum(tf)

    def idf(self, word):

        df = sum([1 for x in self.document_dict if word in self.document_dict[x]])
        #< to smooth or not to smooth, or to smooth, or not to smooth
        if df != 0:
            if self.smooth:
                inverse_df = math.log10(1+(self.documents_n/df))
            else:
                inverse_df = math.log10(self.documents_n/df)
        else:
            return 1 #< !!! identity element, no change in weight

        return inverse_df

#< Read a list of sentences and apply vector addition from context
#< vocabulary = dictionary of words with word_vector and random_vector
#< context = 'CBOW' or 'skipgram', window = window size (default = 1)
class Contexter():
    """
    Reads sentences/text and determines a words context, then performs vector addition.
    Takes pre-weighted vectors

    Contexter.data_info to get the data_info

    PARAMS:
        contexttype: Which type of context, CBOW or skipgram (default: CBOW)
        window: size of context, CBOW: how many words to take, skipgram: how many words to skip (default: 1)
        sentences: set context boundry at sentence boundaries (default: True)
        distance_weights: give weights to words based on distance (default: False) TODO TODO TODO

    INPUT:
        INIT:
            vocabulary of word vectors
        METHODS:
            process_data: sentences/text
            read_contexts: sentence/text
            vector_addittion: word vector, random vector

    OUTPUT:
        Dictionary of words in vocabulary with a updated word_vector
    """
    def __init__(self, vocabulary, contexttype = 'CBOW', window = 1, sentences = True, distance_weights = False):
        self.vocabulary = vocabulary

        self.window = window
        self.context_types = ['CBOW', 'skipgram']
        if contexttype in self.context_types:
            self.contexttype = contexttype
        else:
            self.contexttype = 'CBOW'
        self.distance_weights = distance_weights #TODO add weighting at self.word_addition
        self.sentences = sentences
#        self.history = [] #< ??? history of vector additions, for later use (updating data) NOT IN USE ATM

        self.data_info = {'name': 'Temporary data','context': self.contexttype,'window': self.window, 'weights': 'tf-idf'} #< finetune

    def process_data(self, sentences, update = False):
#        print('Reading text...\n')
        #< read sentence by sentence
        if self.sentences:
            for sentence in sentences:
                self.read_contexts(sentence)
        # read text as one unit
        else:
            self.read_contexts([word for li in sentences for word in li])

        #< needs updating :P :P :P
        #< when updating with new data, re-do all previous vector additions
        if update:
            self.update_contexts()
            #< use history
        else:
            pass
            #< save history or something

        return {x: self.vocabulary[x]['word_vector'] for x in self.vocabulary}

    def read_contexts(self, context_text):
        for i, word in enumerate(context_text):
            context = []

            if self.contexttype == 'CBOW':
                #words before
                if (i-self.window) <= 0:
                    context += context_text[:i]
                else:
                    context += context_text[i-self.window:i]
                #< words after
                if (i+self.window) >= len(context_text):
                    context += context_text[i+1:]
                else:
                    context += context_text[i+1:i+1+self.window]

            elif self.contexttype == 'skipgram':
                #word before
                if (i-self.window-1) < 0:
                    pass
                else:
                    context.append(context_text[i-self.window-1])
                #< words after
                if (i+self.window+1) >= len(context_text):
                    pass
                else:
                    context.append(context_text[i+1+self.window])

            #< add vectors in context
            if context:
                for context_word in context:
                    self.vector_addition(word, context_word)

    def vector_addition(self, word, target_word):
        if word in self.vocabulary.keys() and target_word in self.vocabulary.keys():
            self.vocabulary[word]['word_vector'] += self.vocabulary[target_word]['random_vector']
#            self.history.append((word, target_word))
        else:
            pass

class Similarity():
    """
    Cosine similarities between vectors

    INPUT:
        INIT: vocabulary of words and their word_vectors

        METHODS:
        cosine_similarity:
            input: string1, string2
            output: cosine similarity between str1 and str2

        top:
            input: string
            output: top 5 most cosine similar words
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        #vocab structure {word: [word_vector]}

    def cosine_similarity(self, s_word1, s_word2):
        #< stem input
        word1, word2 = stem(s_word1), stem(s_word2)
#DATAEXTRACTION        word1, word2 = s_word1, s_word2

        #< check if the words exists
        if word1 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word1)
        elif word2 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word2)
        else:
            i_word1 = self.vocabulary[word1]
            i_word2 = self.vocabulary[word2]

        cos_sim = pw.cosine_similarity(i_word1.reshape(1,-1), i_word2.reshape(1,-1))
        return cos_sim[0][0] #self.cosine_measure(i_word1, i_word2)

    #< Dot product
    def dot_product(self, vector1, vector2):
        return sum(map(lambda x: x[0] * x[1], zip(vector1, vector2)))

    #< Cosine similarity
    def cosine_measure(self, vector1, vector2):
        dot_prod = self.dot_product(vector1, vector2)
        vec1 = math.sqrt(self.dot_product(vector1, vector1))
        vec2 = math.sqrt(self.dot_product(vector2, vector2))

        #TODO handle division by zero
        return dot_prod / (vec1 * vec2)

    #TODO: Maybe make prittier somehow? especially make more EFFICIENT! slow with large dataset
    def top_similarity(self, s_word):
        word = stem(s_word)

        if word not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word)
        else:
            word_vec = self.vocabulary[word]

        top = [[0, ""], [0, ""], [0, ""], [0, ""], [0, ""]]

        #< cosine sim between input word and ALL words
        for target_word in self.vocabulary:
            if target_word == word:
                continue

#            cs = self.cosine_measure(word_vec, self.vocabulary[target_word]['word_vector'])
            cs = pw.cosine_similarity(word_vec.reshape(1,-1), self.vocabulary[target_word].reshape(1,-1))

            #< Set highest values
            if cs > top[0][0]:
                top[0][0:] = cs, target_word
            elif cs > top[1][0]:
                top[1][0:] = cs, target_word
            elif cs > top[2][0]:
                top[2][0:] = cs, target_word
            elif cs > top[3][0]:
                top[3][0:] = cs, target_word
            elif cs > top[4][0]:
                top[4][0:] = cs, target_word

        return(top)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class DataOptions():
    """
    Handling data, commands are save, load and info

    save
        input: filename, vocabulary{word: vector}, documents{doc: word_counts}, data_info{info}, weight_setup
        output: filename.npz

    load:
        input: filename
        output:
            vocabulary: defaultdict(dict),
            documents: defaultdict(dict),
            data_info: defaultdict(dict)

    info
        input: optional(string, -weight string, -docs)
        output: info about data or word if supplied
    """
    def __init__(self, vocabulary = None, documents = None, data_info = None, weight_setup = None):
        self.vocabulary = vocabulary
        self.documents = documents
        self.data_info = data_info
        self.data_info['weights'] = weight_setup

    #z Save curretly loaded data
    def save(self, save_name, vocabulary, documents, data_info, weight_setup):
        self.vocabulary = vocabulary
        self.documents = documents
        self.data_info = data_info
        data_info['name'] = save_name
        data_info['weights'] = weight_setup

        try:
            file = '/home/usr1/git/dist_data/d_data/{0}.npz'.format(save_name)
            np.savez(file, vocabulary = vocabulary, documents = documents, data_info = data_info)

            return 'Saved complete: /home/usr1/git/dist_data/d_data/{0}.npz'.format(save_name)

        except Exception as e:
            print(e)

    #< Load a saved datafile
    def load(self, load_name):
        data = np.load('/home/usr1/git/dist_data/d_data/{0}.npz'.format(load_name))

        vocab_npdata = data['vocabulary']
        vocab_npdata.resize(1,1)
        for doc_item in vocab_npdata:
            for defaultd in doc_item:
                self.vocabulary = defaultd

        doc_npdata = data['documents']
        doc_npdata.resize(1,1)
        for doc_item in doc_npdata:
            for defaultd in doc_item:
                self.documents = defaultd

        data_npdata = data['data_info']
        data_npdata.resize(1,1)
        for data_item in data_npdata:
            for defaultd in data_item:
                self.data_info = defaultd

        del data

        return self.vocabulary, self.documents, self.data_info

    def info(self, arg_w = False):
        if self.documents != None and self.data_info != None:
            if arg_w == False:
                return self.documents, self.data_info

            elif arg_w == '-docs':
                return self.documents

            else:
                arg = stem(arg_w)
                if arg not in self.vocabulary.keys():
                    return '{0} does not exist, try again\n'.format(arg)
                else:
                    return {w: self.documents[w][arg] for w in self.documents}, arg

        else:
            print('No data loaded!')

class data_formatter():
    def __init__():
        pass