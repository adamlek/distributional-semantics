# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:21:38 2016
@author: Adam Ek

TODO Come up with some fucking good method/function names?

TODO: distance weights, WHAT WEIGHTS!?
    Ideas?

"""
import sklearn.metrics.pairwise as pw
from stemming.porter2 import stem #ENGLISH
import numpy as np
import numpy.linalg as la

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
        TODO Jesus Christ, New York City => PN [Very aggressive/rough]
    PARAMS:
        preprocess_data:
            pns: convert propernames to PN (default: False)
            nums: convert numbers to NUM (default: True)
            percs: convert percentages to PERC (default: True)
    """
    def __init__(self, docsentences = False, nums = True, perc = True):
        self.vocab = set()
        self.doc_count = defaultdict(dict)

        self.docsentences = docsentences
        self.current_doc = ""
        self.nums = nums
        self.perc = perc

    #< By default create vectors from data
    def preprocess_data(self, filenames):
        """
        input: List of .txt files, params
        output: List of sentences, list of words in vocabulary, dictionary of documents with wordcount in them
        """
        doc_text = defaultdict(str)

        for filename in filenames:
            try:
                with open(filename) as datafile:
#                    print('Reading file {0}...'.format(filename))
                    #< Add dictionary entry with name = filename
                    filen = re.search('(\/+.+\/)(.*.txt)', filename)
                    doc_text[filen.group(2)] = datafile.read()
                    print('Success!\n')

            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                continue

        #TODO add support for inverse sentence frequency
        alltext = []
        for doc in doc_text:
            doc_text[doc] = re.sub('(\.+|\?+|!+)\s{1}', ' . ', doc_text[doc])
            doc_text[doc] = re.sub('[^\sA-ZÅÄÖa-zåäö0-9%-\.]', '', doc_text[doc])
            doc_text[doc] = re.sub('[-\(\)]', ' ', doc_text[doc])
            doc_text[doc] = [doc_text[doc].split()]

            self.current_doc = doc
            self.doc_count[doc] = defaultdict(int)
            for text in doc_text[doc]:
                text = self.sentencizer(text)

                doc_sents = []
                for sentence in text:
                    while '.' in sentence:
                        sentence.remove('.')
                    doc_sents.append(sentence)

                alltext.append(doc_sents)

        #< return as one list or one for each document
        if not self.docsentences:
            alltext = [s for x in alltext for s in x]

        #< cleanup
        del doc_text

        return alltext, self.vocab, self.doc_count

    def format_word(self, word):
        if re.match('(\d+\.\d+)|(\d+\,\d+)|(\d+)', word): #word.isdigit():
            if self.nums:
                word = 'NUM'
        elif '%' in word:
            if self.perc:
                word = 'PERC'
        else:
            word = stem(word)

        if word not in ['NUM', 'PERC', 'STOP']:
            word = word.lower()

        #< populate vocabulary and doc_word_count
        self.vocab.add(word)
        self.doc_count[self.current_doc][word] += 1

        return word

    def sentencizer(self, text):
        sent_end = ['STOP', '.', '!', '?']
        start = [0]
        sentence_text = []
#        pnwords = []

        for i, word in enumerate(text):
            if word != '.':
                text[i] = self.format_word(word)

            if i+1 == len(text):
                sentence_text.append(text[start[0]:])
                break

#            elif word[0].isupper():
#                start.append(i)
#                if text[i+1][0].isupper() and len(start) > 1:
#                    pnwords.append((i, i+1))
#                    print(word, text[i+1])

            if word in sent_end:
                if text[i+1][0].isupper(): #check conditions!!!
                    sentence_text.append(text[start[0]:i+1])
                    start = [i+1]

        return sentence_text


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
#        arr = np.random.random(self.dimensions)

        #< distribute (+1)'s and (-1)'s at random indices
        for i in range(0, self.random_elements):
            if i%2 == 0:
                arr[random.randint(0, self.dimensions-1)] = 1
            else:
                arr[random.randint(0, self.dimensions-1)] = -1

        return arr

class TermRelevance():
    """
    Weights based on tf-idf
    https://en.wikipedia.org/wiki/Tf%E2%80%93idf

    TermRelevance.weight_setup for weight_setup

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
            self.weight_setup += ' log norm'
        elif scheme == 2:
            self.weight_setup += ' double norm'
        else:
            pass

    def weight(self, var, var2=False):
        """
        input: string+vector or list or defaultdict
        output: see individual functions
        """
        if type(var) is str:
            if type(var2) is np.ndarray:
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
                    if (1 + math.log10(self.document_dict[doc][word]))/sum(self.document_dict[doc].values()) > 0:
                        tf.append((1 + math.log10(self.document_dict[doc][word]))/sum(self.document_dict[doc].values()))
                    else:
                        tf.append(0)
                #< Double normalization
                elif self.scheme == 2:
                    max_val = max(self.document_dict[doc].values())
                    w1 = self.document_dict[doc][word]/sum(self.document_dict[doc].values())
                    w2 = max_val*(max_val/sum(self.document_dict[doc].values()))

                    tf.append(0.4 + ((0.4 * w1)/w2))
                #< raw term frequency, (freq/doc_freq)
                else:
                    if self.document_dict[doc][word]/sum(self.document_dict[doc].values()) > 0:
                        tf.append(self.document_dict[doc][word]/sum(self.document_dict[doc].values()))
                    else:
                        tf.append(0)

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
                return math.log10(1+(self.documents_n/df))
            else:
                return math.log10(self.documents_n/df)
        else:
            return 1 #< !!! identity element, no change in weight

class Contexter():
    """
    CURRENTLY: n-skip-2-grams only, i.e. only the steps to skip are customizable

    Reads sentences/text and determines a words context, then performs vector addition.

    Contexter.data_info to get the data_info

    TODO: Add PPMI table: log2( (P(x,y)/P(x)P(y)) )

    PARAMS:
        contexttype: Which type of context, CBOW or skipgram (default: CBOW)
        window: size of context, CBOW: how many words to take, skipgram: how many words to skip (default: 1)
        context_scope: the scope of the contexts, 0 = sentences, 1 = document text, 2 = all of the texts (default: 1)
        distance_weights: give weights to words based on distance (default: False) TODO TODO TODO
        weights: do weighting in this class >>> dict{word: weight}

    INIT:
        vocabulary of word vectors
        >>> dict{word: {word_vector: [word_vector], random_vector: [random_vector]}}
    """
    def __init__(self, vocabulary = False, contexttype = 1, window = 1, context_scope = 2, distance_weights = False, weights = False, readwrite=True, savecontext = False):
        self.vocabulary = vocabulary
        self.vocabt = defaultdict(dict)

        self.contexttype, self.window = contexttype, window

        if self.contexttype == 0:
            context = 'CBOW'
        else:
            self.contexttype = 1
            context = 'skipgram'

        self.distance_weights = distance_weights #TODO add weighting at self.word_addition
        self.context_scope = context_scope
        self.weights = weights

        if not readwrite and not savecontext:
            self.readwrite = True
            self.savecontext = False
        else:
            self.savecontext = savecontext
            self.readwrite = readwrite
        self.data_info = {'name': 'Temporary data','context': context,'window': self.window, 'weights': 'tf-idf'} #< finetune
        self.iters = 0

    def process_data(self, texts, return_vectors = True, docs_count = False, add_pmi=False):
        """
        input: list of sentences
        output: dictionary of {word: updated word_vectors}
        """
        if add_pmi:
            self.readwrite = False
            self.savecontext = True

        if self.context_scope == 0: #< scope is sentences
            for sent in texts:
                if self.savecontext:
                    self.vocabt.update(self.read_contexts(sent))
                else:
                    self.read_contexts(sent)

        elif self.context_scope == 1: #< scope is each document, input: [[doc1],[doc2]]
            for doc in texts:
                self.vocabt.update(self.read_contexts([word for li in doc for word in li]))
        else: #< scope is all of the text, input: [[doc1],[doc2]]
            if self.savecontext:
                self.vocabt = self.read_contexts([word for li in texts for word in li])
            else:
                self.read_contexts([word for li in texts for word in li])

        if docs_count:
            xpmi = self.pmi(docs_count, self.vocabt)

        #< updated vectors or context dictionary readwrite AND savecontext = False?? What happens? Nothing...
        if return_vectors:
            if self.vocabulary:
                if not self.readwrite:
                    for item in self.vocabt:
                        for i, word in enumerate(self.vocabt[item]):
                            if add_pmi:
                                if xpmi:
                                    self.vocabulary[item]['word_vector'] = self.vector_addition(item, word, sum(xpmi[word]))
                            else:
                                self.vocabulary[item]['word_vector'] = self.vector_addition(item, word)

                return {x: self.vocabulary[x]['word_vector'] for x in self.vocabulary}
            else:
                return self.vocabt

        else:
            return self.vocabt

    def read_contexts(self, text):
        """
        input: list of strings
        output: dictionary of {word: [words in context]}
        """
        self.iters += 1
        while '.' in text:
            text.remove('.')

        word_contexts = defaultdict(list)

        for i, item in enumerate(text):
            context = []

            # CBOW
            if self.contexttype == 0:
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

            #< skipgram
            else:
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
                if self.readwrite:
                    for word in context:
                        self.vocabulary[item]['word_vector'] = self.vector_addition(item, word)

                word_contexts[item] += context

        return word_contexts

    def vector_addition(self, word, target_word, ew = False):
        """
        input: string1, string2
        output: word_vector
        """
        if ew:
            return self.vocabulary[word]['word_vector'] + (self.vocabulary[target_word]['random_vector'] * ew)
        else:
            return self.vocabulary[word]['word_vector'] + self.vocabulary[target_word]['random_vector']

    def pmi(self, context_dict, docs_count):
        """
        PMI = log( P((w1, c1)/N) / P(w1/N) * p(c1/N) )

        Get PMI for every word, in relation to the top 100 words.
        1. iterate documents
        2. find top x word counts

        0. new PMIdict { w1...wn: float1...floatn }
        1. iterate context_dict
        2. PMI: w1 => y for y in top x
            a tot(w1+y)/N
            b tot(w1)*tot(y)/N
            c a/b = PMI
            d update PMIdict w1 : y = PMI
        """
        ntop = 1000
        for d in documents:
            #< sum of all words in another words context
            N = sum([1 for x in context_dict for y in context_dict[x]])
            #< Find top n words
            top = set()
            [top.add(y) for x in sorted(docs_count[d].values())[-ntop:] for y in docs_count[d] if docs_count[d][y] == x]

        pmidata = defaultdict(dict)
        for w in context_dict:
            ws1 = sum([1 for x in context_dict for y in context_dict[x] for ww in context_dict[y] if ww == w])
            w1 = ws1/N #< independent probability of w1
            for c in top:
                cs1 = sum([1 for x in context_dict for y in context_dict[x] for ww in context_dict[y] if ww == c])
                c1 = cs1/N #< independent probability of c1
                IP = (w1*c1)
                JP = context_dict[w].count(c1)/N #< join probability of w1 + c1
                if JP == 0:
                    continue
                elif IP == 0:
                    continue

                pmidata[w][c] = math.log10(JP/IP)

        return pmidata

class Similarity():
    """
    Cosine similarities between vectors

    INIT:
        vocabulary of words and their word_vectors
        >>> dict{word:[word_vector]}
    """
    def __init__(self, vocabulary, pmi = False):
        self.vocabulary = vocabulary
        #vocab structure {word: [word_vector]}

        self.pmi = pmi

    def norm(self, v):
        tot = sum(v)
        return np.array([round(x/tot) for x in v])

    def cosine_similarity(self, s_word1, s_word2, pmi = False):
        """
        input: string1, string2
        output: cosine similarity
        """
        word1, word2 = stem(s_word1), stem(s_word2)
        #< check if the words exists
        if word1 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word1)
        elif word2 not in self.vocabulary.keys():
            return '{0} does not exist, try again\n'.format(s_word2)
        else:
            i_word1 = self.vocabulary[word1]
            i_word2 = self.vocabulary[word2]

#        if self.pmi:
#            i_word1 *= sum(self.pmi[word1].values())
#            i_word2 *= sum(self.pmi[word2].values())

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

        #TODO handle division by zero ???
        if vec1 * vec2 <= 0:
            return 0

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