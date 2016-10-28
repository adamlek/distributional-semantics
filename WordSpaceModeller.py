# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:21:38 2016
@author: Adam Ek

TODO Come up with some fucking good method/function names?

TODO Check stemmer!!!
TODO HANDLE: Proper names =>>> PN

TODO: distance weights, WHAT WEIGHTS!?
    Ideas?

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
    Reads data from .txt files and organizes them into sentence, creates a vocabulary and summarises word counts in each document.
    Categorizations:
        1, 1.2, 1/3, 1950 => NUM
        1%, 1.33% => PERC
        Jesus Christ, New York City => PN [Very aggressive/rough]

    PARAMS:
        preprocess_data:
            pns: convert propernames to PN (default: True)
            nums: convert numbers to NUM (default: True)
            percs: convert percentages to PERC (default: True)
    """
    def __init__(self):
        self.vocabulary = []
        self.documents = defaultdict(dict)
        self.current_doc = ""
        self.pns = False
        self.nums = True
        self.percs = True

    #< By default create vectors from data
    def preprocess_data(self, filenames, pns = False, nums = True, perc = True):
        """
        input: List of .txt files, params
        output: List of sentences, list of words in vocabulary, dictionary of documents with wordcount in them
        """
        self.pns, self.nums, self.percs = pns, nums, perc
        sentences_collection = []
#        doc_text = []

        for filename in filenames:
            try:
                with open(filename) as datafile:
                    print('Reading file {0}...'.format(filename))
                    #< Add dictionary entry with name = filename
                    filen = re.search('(\/+.+\/)(.*.txt)', filename)
                    self.current_doc = filen.group(2)
                    self.documents[filen.group(2)] = defaultdict(int)

                    for line in datafile:
#                        line = re.sub('(\[\d+\])', '', line)
                        #< separate line into sentences
                        sentences, conc = self.sentencizer(line.rstrip())
                        for i, sentence in enumerate(sentences):
                            if sentence:
                                formatted_sentence = []
                                for word in sentence:
                                    #< format word
                                    #< if self-preservation => self, preservation
                                    if '-' in word:
                                        words = word.split('-')
                                        for wordn in words:
                                            formatted_sentence.append(self.word_formatter(wordn))
                                        continue
                                    formatted_sentence.append(self.word_formatter(word))

                                while '' in formatted_sentence:
                                    formatted_sentence.remove('')

                                #< add sentence to last sentence
                                if i == conc:
                                    sentences_collection[-1] += formatted_sentence
                                else:
                                    sentences_collection.append(formatted_sentence)

                    print('Success!\n')
#                doc_text.append(sentences_collection)

            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                continue

        return sentences_collection, self.vocabulary, self.documents

    #TODO fix option to apply PN to propernames
    #< Create sentences from a line in a document
    def sentencizer(self, line):
        """
        input: string
        output: list of sentences
        """
        start_sent = []
        sentences = []
        addtolast = None
        if line:
            if line[0].islower():
                start_sent.append(0)
                addtolast = 0

            for i, symbol in enumerate(line):
                #< capture everything not captured, then break
                if i+2 >= len(line):
                    if self.pns:
                        if start_sent:
                            sentences.append(self.propernamer(line[start_sent[0]:].split()))
                        else:
                            sentences.append(line.split())
                    else:
                        sentences.append(line.split())
                    break

                #< add index of uppercase symbols
                elif symbol.isupper():
                    start_sent.append(i)

                #< . ? or ! and i+2 isupper, sentence end
                elif symbol == '.' or symbol == '?' or symbol == '!':
                    if i > 2:
                        if line[i+2].isupper():
                            #< needs some fixing/finetuning
                            if line[i-3:i-2].islower() or line[i-3:i-2].isdigit(): #exclude Sir. Mr. Mrs. etc
                                if start_sent:
                                    if self.pns:
                                        sentences.append(self.propernamer(line[start_sent[0]:i].split()))
                                    else:
                                        sentences.append(line[start_sent[0]:i].split())
                                start_sent = []

        return sentences, addtolast

    def propernamer(self, sent):
        """
        input: list of strings
        output: list of strings
        """
        for i, word in enumerate(sent):
            if i != 0: #< Do not check first word. "All Adams are something." => "PN are something."
                if word[0].isupper():
                    if i+1 != len(sent):
                        if sent[i+1][0].isupper():
                            del sent[i]
                            del sent[i]
                            if i != len(sent):
                                if sent[i][0].isupper():
                                    del sent[i]

                            sent.insert(i, 'PN')
        return sent

    #< Format words
    #< word: self-preservation => self preservation
    #< nums: 5-6 => 56 => NUM, 3.1223 => 31223 => NUM
    def word_formatter(self, word):
        """
        input: string
        output: string
        """
        #TODO 1950s, 13th
        #< remove special things inside words
        word = re.sub('[^A-ZÅÄÖa-zåäö0-9%]', '', word)

        #< stem and replace word
        word = stem(word)

        #< dont add null words
        if word == '':
            return ''

        #< FINE TUNE DATA
        if self.nums:
            if word.isdigit():
                word = 'NUM'
        #< 12% etc => PERC
        if self.percs:
            if '%' in word:
                word = 'PERC'

        if word not in ['NUM','PERC','PN']: #alt: word.isupper():
            word = word.lower()

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
        dimensions: dimensionality of the random/word vector (default: 1024)
        random_elements: how many random indices to insert +1's and -1's into in the random vector (default: 6)
    """
    def __init__(self, dimensions = 1024, random_elements = 4):
        self.dimensions = dimensions
        self.random_elements = random_elements
        self.vocabulary = defaultdict(dict)

    def vocabulary_vectorizer(self, word_list):
        """
        input: list of words
        output: dict{word:{word_vector:[], random_vector:[]}}
        """
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

        scheme 1: log normalization
            1 + log10(freq/doc_freq) * log(N/n)

        scheme 2: double normalization
            0.5 + (0.5 * ((freq/doc_freq)/(max_freq*(max_freq/doc_freq)))) * log(N/n)

    PARAMS:
        scheme: select a weighting scheme to use (default: 0)
        smooth_idf: smooth the idf weight log(1+(N/n))(default: False)
        doidf: compute idf (default: True)

    INIT:
        document_dict: dictionary of documents and their word count
        >>> dict{doc{word: count}}
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

    def weight(self, var, var2=False):
        """
        input: string+vector or list or defaultdict
        output: see individual functions
        """
        if type(var) is str:
            return self.weight_vector(var, var2)
        elif type(var) is list:
            return self.weight_list(var)
        elif type(var) is defaultdict:
            return self.weight_dict(var)
        else:
            return None

    def weight_dict(self, vocab):
        """
        input: dict{word: {word_vector:[], random_vector:[]}}
        output: dict{word: {word_vector:[], random_vector:[]}}
        """
        for word in vocab:
            vocab[word]['random_vector'] = self.weight_vector(word, vocab[word]['random_vector'])

        return vocab

    def weight_vector(self, word, vector):
        """
        input: string, vector
        output: vector
        """
        weight = self.tf(word)
        if self.documents_n > 1:
            if self.doidf:
                weight *= self.idf(word)

        #< save weights for lookup
        self.word_weights[word] = weight
        #< return vector * tf-idf, save tfidf value for future info???
        return vector * weight

    def weight_list(self, wordlist):
        """
        input: list of strings
        output: dict{word:weight}
        """
        for word in wordlist:
            weight = self.tf(word)
            if self.documents_n > 1:
                if self.doidf:
                    weight *= self.idf(word)

            self.word_weights[word] = weight

        return self.word_weights

    def tf(self, word):
        """
        input: string
        output: float
        """
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
        """
        input: string
        output: float
        """
        df = sum([1 for x in self.document_dict if word in self.document_dict[x]])
        #< to smooth or not to smooth, or to smooth, or not to smooth
        if df != 0:
            if self.smooth_idf:
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

    Contexter.data_info to get the data_info

    PARAMS:
        contexttype: Which type of context, CBOW or skipgram (default: CBOW)
        window: size of context, CBOW: how many words to take, skipgram: how many words to skip (default: 1)
        sentences: set context boundry at sentence boundaries (default: True)
        distance_weights: give weights to words based on distance (default: False) TODO TODO TODO
        weights: do weighting in this class >>> dict{word: weight}

    INIT:
        vocabulary of word vectors
        >>> dict{word: {word_vector: [word_vector], random_vector: [random_vector]}}
    """
    def __init__(self, vocabulary, contexttype = 'CBOW', window = 1, sentences = False, distance_weights = False, weights = False):
        self.vocabulary = vocabulary

        self.window = window
        self.context_types = ['CBOW', 'skipgram']
        if contexttype in self.context_types:
            self.contexttype = contexttype
        else:
            self.contexttype = 'CBOW'

        self.distance_weights = distance_weights #TODO add weighting at self.word_addition
        self.sentences = sentences

        self.weights = weights

        self.data_info = {'name': 'Temporary data','context': self.contexttype,'window': self.window, 'weights': 'tf-idf'} #< finetune


    def process_data(self, texts, update = False):
        """
        input: list of sentences
        output: dictionary of {word: updated word_vectors}
        """

        vocabt = defaultdict(dict)

        if self.sentences:
            for sent in texts:
                vocabt.update(self.read_contexts(sent))
        else:
            vocabt = self.read_contexts([word for li in texts for word in li])

        for item in vocabt:
            for i, word in enumerate(vocabt[item]):
                self.vocabulary[item]['word_vector'] = self.vector_addition(item, word)
#                if item == 'the':
#                    if i < 50:
#                        print(item, word)

        return {x: self.vocabulary[x]['word_vector'] for x in self.vocabulary}

    #< only requires text, will always output contexts of every word
    def read_contexts(self, text):
        """
        input: list of strings or list of lists[strings]
        output: dictionary of {word: [words in context]}
        """
        word_contexts = defaultdict(list)

        for i, item in enumerate(text):
                context = []
                if self.contexttype == 'CBOW':
                    #words before
                    if (i-self.window) <= 0:
                        context += text[:i]
                    else:
                        context += text[i-self.window:i]
                    #< words after
                    if (i+self.window) >= len(text):
                        context += text[i+1:]
                    else:
                        context += text[i+1:i+1+self.window]

                elif self.contexttype == 'skipgram':
                    #word before
                    if (i-self.window-1) < 0:
                        pass
                    else:
                        context.append(text[i-self.window-1])
                    #< words after
                    if (i+self.window+1) >= len(text):
                        pass
                    else:
                        context.append(text[i+1+self.window])

                if context:
                    word_contexts[item] += context

        return word_contexts

    def vector_addition(self, word, target_word):
        """
        input: string1, string2
        output: word_vector
        """
        if word in self.vocabulary.keys() and target_word in self.vocabulary.keys():
            if self.weights:
                return self.vocabulary[word]['word_vector'] + (self.vocabulary[target_word]['random_vector'] * self.weights[word])
            else:
                return self.vocabulary[word]['word_vector'] + self.vocabulary[target_word]['random_vector']
        else: #TODO check for words that does not exist
            return self.vocabulary[word]['word_vector']


class Similarity():
    """
    Cosine similarities between vectors

    INIT:
        vocabulary of words and their word_vectors
        >>> dict{word:[word_vector]}
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        #vocab structure {word: [word_vector]}

    def cosine_similarity(self, s_word1, s_word2):
        """
        input: string1, string2
        output: cosine similarity
        """
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

#        print(i_word1)
        cos_sim = pw.cosine_similarity(i_word1.reshape(1,-1), i_word2.reshape(1,-1))
        return cos_sim[0][0] #self.cosine_measure(i_word1, i_word2)

    #TODO: Maybe make prittier somehow? especially make more EFFICIENT! slow with large dataset
    def top_similarity(self, s_word):
        """
        input: string
        output: top 5 most cosine similar words
        """
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

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class DataOptions():
    """
    Handling data, commands are save, load and info
    """
    def __init__(self, vocabulary = [], documents = defaultdict(dict), data_info = defaultdict(str), weight_setup = None):
        self.vocabulary = vocabulary
        self.documents = documents
        self.data_info = data_info
        self.data_info['weights'] = weight_setup

    #z Save curretly loaded data
    def save(self, save_name, vocabulary, documents, data_info, weight_setup):
        """
        input: filename, vocabulary{word: vector}, documents{doc: word_counts}, data_info{name, context, window, weights}, weight_setup
        output: filename.npz
        """
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
        """
        input: filename
        output:
            vocabulary: defaultdict(dict),
            documents: defaultdict(dict),
            data_info: defaultdict(dict)
        """
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
        """
        input: optional(string, -weight string, -docs)
        output: info about data or word if supplied
        """
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

class TextDocFormatter():
    def __init__(self):
        pass

    def read_texts(self, doc_path):
        with open(doc_path) as txt_file:
            pass

        pass