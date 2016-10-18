# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:21:38 2016

@author: Adam Ek

"""

import numpy as np
from collections import defaultdict
import re
import random
import math
import sklearn.metrics.pairwise as pw
from stemming.porter2 import stem #ENGLISH


class RandomIndexing():
    """
    create_vectors = True
        input: .txt file
        output:
            success rate,
            [sentences],
            { vocabulary: word: {word_vector, random_vector, word_count}, documents: name: {words, unique_words}, data_info: None }

    create_vectors = False
        input: .txt file
        output: success_rate, [sentences]

    """
    def __init__(self):
        #< word: word vector, random vector, word counts
        self.vocabulary = defaultdict(dict)
        #< document: word counts
        self.documents = defaultdict(dict)
        #< data info
        self.data_info = defaultdict(dict)
        self.current_doc = ''

    #< By default create vectors from data
    def process_data(self, filenames, create_vectors = True):
        print('Processing data...')
        success_rate = [0,0]
        sentences = []
        for filename in filenames:
            success_rate[1] += 1
            try:
                with open(filename) as training:
                    print('Reading file {0}...'.format(filename))
                    filen = re.search('(\w*).txt', filename)
                    self.documents[filen.group(0)] = {'words': 0, 'unique_words': 0 }
                    self.current_doc = filen.group(0)
                    for row in training:
                         #< separate row into sentences
                         row = row.strip().split('. ')
                         for sentence in row:
                            sentence = sentence.split()
                            #< format sentence
                            formatted_sentence = [w.strip().lower() for w in sentence]
                            if formatted_sentence:
                                formatted_sentence = self.sentence_formatter(formatted_sentence, create_vectors)
                                sentences.append(formatted_sentence)
                    #< Print message to user
                    success_rate[0] += 1
                    print('Success!\n')

            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                continue
            
        if create_vectors:
            data = {'vocabulary': self.vocabulary, 'documents': self.documents, 'data_info': None}
            return sentences, data
        else:
            return sentences, None


    def sentence_formatter(self, sentence, vectorize):
        upd_sentence = []

        for i, word in enumerate(sentence):
            #< word: self-preservation => selfpreservation
            #< nums: 5-6 => 56 => NUM, 3.1223 => 31223 => NUM

            #< remove special things inside words
            word = re.sub('[^a-zåäö0-9%]', '', word)

            #< stem and replace word
            word = stem(word)

            #< dont add null words
            if word == '':
                continue

            #< FINE TUNE DATA
            #< change numbers to NUM
            if word.isdigit():
                word = 'NUM'
            #< 12% etc => PERC
            elif '%' in word:
                word = 'PERC'

            upd_sentence.append(word)
            
            if vectorize:
                self.documents[self.current_doc]['words'] += 1                
                
                #< set up random_vectors
                if word not in self.vocabulary.keys():
                    self.vocabulary[word]['word_vector'] = np.zeros(1024)
                    self.vocabulary[word]['random_vector'] = self.random_vector()
                    self.vocabulary[word]['word_count'] = [0]*len(self.documents)
                    self.vocabulary[word]['word_count'][-1] = 1
                    
                    self.documents[self.current_doc]['unique_words'] += 1

                else:
                    #< expand document word_count list if necessary
                    if len(self.vocabulary[word]['word_count']) != len(self.documents):
                        for i in range(0, (len(self.documents)-len(self.vocabulary[word]['word_count']))):
                            self.vocabulary[word]['word_count'].append(0)

                    #< update document word count
                    self.vocabulary[word]['word_count'][-1] +=1

        return upd_sentence

    def random_vector(self):
        arr = np.zeros(1024)

        #< distribute (+1)'s and (-1)'s at random indices
        for i in range(0, 6):
            if i%2 == 0:
                arr[random.randint(0, 1023)] = 1
            else:
                arr[random.randint(0, 1023)] = -1

        return arr


#< apply weights to vectors
class Weighter():
    """
    class init: 
        scheme (of no consequence atm)    
        wc_doc = [word count in doc_n]
    
    weight:
        input:
            vector,
            [word_count in doc_n, ...],
        output:
            random_vector
    """
    def __init__(self, scheme, wc_doc):
        self.scheme = scheme
        self.wc_doc = wc_doc
#        self.schemes = {'tf-idf': sum(x)/len(d) * math.log(sum(d)/len([z for z in x if z != 0]))}

    def weight(self, random_vector, count, docs):
        
        #calculate tf and idf
        tf = sum([1 + math.log10(x/y) for x, y in zip(count, self.wc_doc) if x != 0])
        idf = math.log10(len(self.wc_doc)/len([x for x in count if x != 0]))
                
        #< return vector * tf-idf
        return random_vector * (tf * idf)



class Contexts():
    """
    input:
        data: vocabulary: word: {word_vector, random_vector, (word_counts)}
        context: CBOW or skipgram
        window: integer

    output:
        vocabulary: word: {word_vector, random_vector, (word_counts)},
        data_info: {name, context, window, weights}
    """
    def __init__(self, vocabulary, context, window):
        self.vocabulary = vocabulary

        self.window = 1
        self.context_types = {'CBOW': 0, 'skipgram': 1}
        if context in self.context_types.keys():
            self.context = context
        else:
            self.context = 'CBOW'

    """
    input: list of sentences
    output: vocabulary, data_info
    """
    def read_data(self, sentences, update = False):
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
            for n in range(1,self.window+1):
                #< words before
                if i - n >= 0: #< exclude negative indexes
                    try:
                        prev_word = sentence[i-n-self.context_types[self.context]]
                        self.vector_addition(word, prev_word)
                    except:
                        pass

                #< words after
                if i + n != len(sentence):
                    try:
                        next_word = sentence[i+n+self.context_types[self.context]]
                        self.vector_addition(word, next_word)
                    except:
                        pass

    def vector_addition(self, word, target_word):
        if word in self.vocabulary.keys() and target_word in self.vocabulary.keys():
            self.vocabulary[word]['word_vector'] += self.vocabulary[target_word]['random_vector']
        else:
            pass


class Similarity():
    """
    class init: vocabulary: word: 'word_vector': vector
    
    command: sim word1 word2
    input: word1, word2
    output: cosine similarity between word1 and word2
    
    command: top word
    input: word
    output: top 5 similar words
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def cosine_similarity(self, s_word1, s_word2):
        #< stem input
        word1, word2 = stem(s_word1), stem(s_word2)

        #< check if the words exists
        if word1 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word1)
        elif word2 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word2)
        else:
            i_word1 = self.vocabulary[word1]['word_vector']
            i_word2 = self.vocabulary[word2]['word_vector']

        return self.cosine_measure(i_word1, i_word2)
        
    #< Dot product
    def dot(self, vector1, vector2):
        return sum(map(lambda x: x[0] * x[1], zip(vector1, vector2)))
    
    #< Cosine similarity
    def cosine_measure(self, vector1, vector2):        
        vec1 = math.sqrt(self.dot(vector1, vector1))
        vec2 = math.sqrt(self.dot(vector2, vector2))
        
        return self.dot(vector1, vector2) / (vec1 * vec2)

    def top(self, s_word):
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
    command: save filename
    input: filename    
    output: filename.npz
    
    command: load filename
    input: filename
    output: vocabulary: defaultdict(dict), defaultdict(dict), defaultdict(dict)
    
    command: info
    input: optional(word)
    output: info about data or word if supplied
    """
    def __init__(self, vocabulary = False, documents = False, data_info = False):
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
        for vocab_item in vocab_npdata:
            for defaultd in vocab_item:
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

        return {'vocabulary': self.vocabulary, 'documents': self.documents, 'data_info': self.data_info}


    def info(self, arg_w = False):
        if arg_w == False:
            print('Data info:')
            print('Name: {0}'.format(self.data_info['name']))
            print('Weighting scheme: {0}'.format(self.data_info['weights']))
            print('Context type: {0}'.format(self.data_info['context']))
            print('Context window size: {0}'.format(self.data_info['window']))
            print('Unique words: {0}'.format(sum([self.documents[x]['unique_words'] for x in self.documents])))
            print('Total words: {0}\n'.format(sum([self.documents[x]['words'] for x in self.documents])))
            
            print('Document info:')
            for doc_info in self.documents:
                print(doc_info)
                print('Unique words: {0}'.format(self.documents[doc_info]['unique_words']))
                print('Total words: {0}\n'.format(self.documents[doc_info]['words']))
            
        else:
            arg = stem(arg_w)
            if arg not in self.vocabulary.keys():
                return '{0} does not exist, try again\n'.format(arg)
            else:
                word_data = self.vocabulary[arg]
                print('"{0}" stemmed to "{1}"\n'.format(arg_w, arg))

                print('{0} occurences in {1} documents'.format(sum(word_data['word_count']), len([x for x in word_data['word_count'] if x != 0])))

class Extender():
    def extender(item, count, elements = 0):
        pass
        
    
    
    
    
    
    