# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:21:38 2016

@author: Adam Ek

TODO: fix descriptions
TODO: name change top WordSpaceModeller.py
"""

"""
PARAMS:
    P:
INPUT:
    N:
OUTPUT:
    I:
"""
import sklearn.metrics.pairwise as pw
from stemming.porter2 import stem #ENGLISH
import numpy as np

from collections import defaultdict
import re
import random
import math



class DataReader():

    def __init__(self):
        self.vocabulary = []
        self.documents = defaultdict(dict)
        self.current_doc = ""

        #< ??? update...
        self.loaded_data = ''


    #< By default create vectors from data
    def preprocess_data(self, filenames, read_only = False, numerize = False):
        """
        PARAMS:
            Numerize: NULL ATM
            read_only = default(False): Only output a list of sentences
        INPUT:
            List of .txt files
        OUTPUT:
            List of sentences, list of words in vocabulary, dictionary of documents with wordcount in them
        """
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
                                    word = self.word_formatter(word)
                                    if word:
                                        formatted_sentence.append(word)

                                    if read_only:
                                        continue
                                    else: #TODO !!! Make prittier? maybe vocab as a set?
                                        if word not in self.vocabulary:
                                            self.vocabulary.append(word)
                                            self.documents[self.current_doc][word] = 1
                                        else:
                                            #< update word count in document
                                            self.documents[self.current_doc][word] +=1

                                sentences_collection.append(formatted_sentence)
                    print('Success!\n')

            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                continue

        if read_only:
            return sentences_collection
        else:
            return sentences_collection, self.vocabulary, self.documents


    #< Create sentences from a line in a document
    #< "U.S.A. is blaaaa"
    def sentencizer(self, line):
        start_sent = 0
        sentences = []

        for i, symbol in enumerate(line):
            if symbol.isupper():
                if line[i+1] != '.':
                    start_sent = i

            if i+2 >= len(line):
                sentences.append(line[start_sent:-1].lower().split())
                break

            elif symbol == '.' or symbol == '?' or symbol == '!':
                if line[i+2].isupper():
                    sentences.append(line[start_sent:i].lower().split())

        return sentences

    #< Format words
    #< word: self-preservation => selfpreservation
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

        return word


class WSModel():
    def __init__(self, dimensions = 1024, random_elements = 6, ri_model = True):
        self.ri_model = ri_model
        self.dimensions = dimensions
        self.random_elements = random_elements
        self.vocabulary = defaultdict(dict)

    def vocabulary_vectorizer(self, word_list):
        """
        TODO: ???!!!!!! output format, only give random_vector if ri_model (create word_vectors at sentence reading)

        INPUT:
            List of words
        OUTPUT:
            Dictionary of words in vocabulary with a word_vector and a random_vector
        """
        for word in word_list:
            self.vocabulary[word]['word_vector'] = np.zeros(self.dimensions)
            if self.ri_model:
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


#< document_count = dictionary of documents with word_count of words in them
class Weighter():
    def __init__(self, scheme, documents_count):
        self.scheme = scheme
        self.documents_count = documents_count
        self.documents_n = len(documents_count)
#        self.schemes = {'tf-idf': sum(x)/len(d) * math.log(sum(d)/len([z for z in x if z != 0]))}

    def weight(self, word, vector):
        """
        INPUT:
            Word, vector
        OUTPUT:
            tf-idf weighted vector
        """
        #< calculate tf and idf
        tf = []
        df = 0
        for doc in self.documents_count:
            if word in self.documents_count[doc]:
                df += 1
                tf.append(1 + math.log10(self.documents_count[doc][word]/sum(self.documents_count[doc].values())))

        #< return vector * tf-idf
        return vector * (sum(tf) * math.log10(self.documents_n/df))


#< Read a list of sentences and apply vector addition from context
#< vocabulary = dictionary of words with word_vector and random_vector
#< context = 'CBOW' or 'skipgram', window = window size (default = 1)
class Contexter():
    def __init__(self, vocabulary, context = 'CBOW', window = 1):
        self.vocabulary = vocabulary

        if window <= 5:
            self.window = window
        else:
            window = 5

        self.context_types = {'CBOW': 0, 'skipgram': 1}
        if context in self.context_types.keys():
            self.context = context
        else:
            self.context = 'CBOW'


    def process_data(self, sentences, update = False):
        """
        INPUT:
            list of sentences
        OUTPUT:
            Dictionary of words in vocabulary with a word_vector and a random_vector
            Dictionary of data_info: name: x, context: y, window: n, weights: m
        """

        print('Reading sentences...\n')
        #< read all sentences
        for sentence in sentences:
            self.read_contexts(sentence)

        #< when updating with new data, re-do all previous vector additions
        if update:
            self.update_contexts()

        return self.vocabulary, {'name': 'Temporary data',
                                  'context': self.context,
                                  'window': self.window,
                                  'weights': 'tf-idf'}

    def read_contexts(self, sentence):
        for i, word in enumerate(sentence):
            context = []
            #TODO !!! words before index, fix skip-gram(???)
            if (i-n-self.context_types[self.context]) >= 0 or i != 0:
                context += sentence[i-n-self.context_types[self.context]:i-1] #words before

            #TODO !!! words after index, fix skip-gram(???)
            if (i+n+self.context_types[self.context]) <= len(sentence):
                context += sentence[i:i+n+self.context_types[self.context]] #words after

            if context:
                for context_word in context:
                    self.vector_addition(word, context_word)

    #TODO: Add history
    def vector_addition(self, word, target_word):
        if word in self.vocabulary.keys() and target_word in self.vocabulary.keys():
            self.vocabulary[word]['word_vector'] += self.vocabulary[target_word]['random_vector']
        else:
            pass


class Similarity():
    """
    INPUT:
        dictionary of words with word_vectors

    cosine_similarity:
        input:
            str1, str2
        output:
            cosine similarity between str1 and str2

    top:
        input:
            str
        output:
            top 5 most cosine similar words
    """
    def __init__(self, vocabulary):
        #TODO ??? init datatype, only take word_vectors
        self.vocabulary = vocabulary


    def cosine_similarity(self, s_word1, s_word2):
        #< stem input

#TEMP        word1, word2 = stem(s_word1), stem(s_word2)
        word1, word2 = s_word1, s_word2

        #< check if the words exists
        if word1 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word1)
        elif word2 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word2)
        else:
            i_word1 = self.vocabulary[word1]['word_vector']
            i_word2 = self.vocabulary[word2]['word_vector']

        cos_sim = pw.cosine_similarity(i_word1.reshape(1,-1), i_word2.reshape(1,-1))
        return cos_sim[0][0] #self.cosine_measure(i_word1, i_word2)

    #< Dot product
    #TODO !!! Add to cosine_measure
    def dot_product(self, vector1, vector2):
        return sum(map(lambda x: x[0] * x[1], zip(vector1, vector2)))

    #< Cosine similarity
    def cosine_measure(self, vector1, vector2):
        dot_prod = self.dot_product(vector1, vector2)
        vec1 = math.sqrt(self.dot_product(vector1, vector1))
        vec2 = math.sqrt(self.dot_product(vector2, vector2))

        return dot_prod / (vec1 * vec2)

    def top_similarity(self, s_word):
        word = stem(s_word)

        if word not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word)
        else:
            word_vec = self.vocabulary[word]['word_vector']

        top = [[0, ""], [0, ""], [0, ""], [0, ""], [0, ""]]

        #< cosine sim between input word and ALL words
        for target_word in self.vocabulary:
            if target_word == word:
                continue

#            cs = self.cosine_measure(word_vec, self.vocabulary[target_word]['word_vector'])
            cs = pw.cosine_similarity(word_vec.reshape(1,-1), self.vocabulary[target_word]['word_vector'].reshape(1,-1))

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
    save
        input:
            filename
        output:
            filename.npz

    load:
        input:
            filename
        output:
            vocabulary: defaultdict(dict),
            documents: defaultdict(dict),
            data_info: defaultdict(dict)

    info
        input:
            optional(word)
        output:
            info about data or word if supplied
    """
    def __init__(self, vocabulary = None, documents = None, data_info = None):
        self.vocabulary = vocabulary
        self.documents = documents
        self.data_info = data_info

    #z Save curretly loaded data
    def save(self, save_name, vocabulary, documents, data_info):

        self.vocabulary = vocabulary
        self.documents = documents
        self.data_info = data_info

        try:
            file = '/home/usr1/git/dist_data/d_data/{0}.npz'.format(save_name)
            data_info['name'] = save_name
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
                print(self.vocabulary)

        doc_npdata = data['documents']
        doc_npdata.resize(1,1)
        for doc_item in doc_npdata:
            for defaultd in doc_item:
                self.documents = defaultd
                print(self.documents)

        data_npdata = data['data_info']
        data_npdata.resize(1,1)
        for data_item in data_npdata:
            for defaultd in data_item:
                self.data_info = defaultd

        del data

        return self.vocabulary, self.documents, self.data_info

    #TODO add option to show documents or not
    def info(self, arg_w = False):
        if self.documents != None and self.data_info != None:
            if arg_w == False:
                print('Data info:')
                print('Name: {0}'.format(self.data_info['name']))
                print('Weighting scheme: {0}'.format(self.data_info['weights']))
                print('Context type: {0}'.format(self.data_info['context']))
                print('Context window size: {0}'.format(self.data_info['window']))
                print('Unique words: {0}'.format(sum([len(self.documents[x].keys()) for x in self.documents])))
                print('Total words: {0}\n'.format(sum([sum(self.documents[x].values()) for x in self.documents])))

                print('Document info:')
                for doc_info in self.documents:
                    print(doc_info)
                    print('Unique words: {0}'.format(len(self.documents[doc_info].keys())))
                    print('Total words: {0}\n'.format(sum(self.documents[doc_info].values())))

            else:
                arg = stem(arg_w)
                if arg not in self.vocabulary.keys():
                    return '{0} does not exist, try again\n'.format(arg)
                else:
                    print('"{0}" stemmed to "{1}"\n'.format(arg_w, arg))
                    total = [0, 0]
                    for w in self.documents:
                        if arg in self.documents[w]:
                            print('{0}:\n {1} occurrences\n'.format(w, self.documents[w][arg]))
                            total[0] += self.documents[w][arg]
                            total[1] += 1
                    print('{0} occurences in {1} documents'.format(total[0], total[1]))
        else:
            print('No data loaded!')


class data_formatter():
    def __init__():
        pass






