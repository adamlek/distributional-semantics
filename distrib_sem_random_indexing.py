# -*- coding: utf-8 -*-
"""
----------------------------------------
Created on Thu Sep 22 15:08:36 2016

@author: Adam Ek
----------------------------------------

Random indexing:

    (1): Generate index vectors
        gather all the words in the data:
            for each word, generate a d-dimensional matrix of 0's with a small amount of (+1) and (-1) distributed

    (2):
        scan text
            each time a word w occurs in a context, add that contexts index vector to the vector of that word

TO ADD:

"""
import numpy as np
from numpy import linalg as la
import re
import random
import math
import sklearn.metrics.pairwise as pw
import matplotlib.pyplot as plt
from stemming.porter2 import stem #ENGLISH
import os.path
import sys

class DistributionalSemantics():

    def __init__(self):
        self.vocabulary = []
        self.index_vectors = []
        self.word_vectors = []

        self.weights = []
        self.word_count = []
        self.documents = 0
        self.total_words = []

        self.superlist = []
        self.evaled_data = []
        #< Current instance, changed when loading/saving
        self.current_load = None
        #<'CBOW': [WORD-BEHIND, WORD, WORDAFTER] or 'skip-gram': [WORD-BEFORE, skip-word, WORD, skip-word, WORD-AFTER]
        self.context_type = 'CBOW' 

        self.window = 1 #< how many words before/after to consider being a part of the context

    #< create an index vector for each word
    def rand_index_vector(self):
        arr = np.zeros(1024)

        #< distribute (+1)'s and (-1)'s at random indices
        for i in range(0, 4):
            if i%2 == 0:
                arr[random.randint(0, 1023)] = 1
            else:
                arr[random.randint(0, 1023)] = -1

        return arr

    #< Read sentences from data file
    def vectorizer(self, formatted_sentence):
        for i, word in enumerate(formatted_sentence):
            #< word: self-preservation => selfpreservation
            #< nums: 5-6 => 56 => NUM, 3.1223 => 31223 => NUM
            #< remove special things inside words
            formatted_sentence[i] = re.sub('[^a-zåäö0-9%]', '', formatted_sentence[i])
            word = re.sub('[^a-zåäö0-9%]', '', word)

            #< stem and replace word
            word = stem(word)
            formatted_sentence[i] = stem(formatted_sentence[i])

            #< dont add null words
            if word == '':
                continue

            self.total_words[self.documents-1] += 1

            #< FINE TUNE DATA
            #< change numbers to NUM
            if word.lstrip('-').isdigit():
                word = 'NUM'
                formatted_sentence[i] = 'NUM'
            #< 12% etc => PERC
            elif '%' in word:
                word = 'PERC'
                formatted_sentence[i] = 'PERC'

            #< set up vectors
            if word not in self.vocabulary:
                self.vocabulary.append(word)
                self.index_vectors.append(self.rand_index_vector())
                self.word_vectors.append(np.zeros(1024))
                self.weights.append([0, int(self.documents)])
                self.word_count.append([0]*int(self.documents))
                self.word_count[-1][-1] = 1

            #< create weight tools
            else:
                word_id = self.vocabulary.index(word)
                
                #< add document occurance 
                if int(self.documents) not in self.weights[word_id][1:]:
                    self.weights[word_id] = np.concatenate((self.weights[word_id],  [int(self.documents)]))

                #< add word count for each document
                if len(self.word_count[word_id]) != int(self.documents):
                    for z in range(0,(int(self.documents)-len(self.word_count[word_id]))):
                        self.word_count[word_id] = np.concatenate((self.word_count[word_id],  [0]))

                self.word_count[word_id][int(self.documents)-1] += 1

    def process_data(self, filenames):
        print('Processing data...')
        success_rate = [0,0]
        for filename in filenames:
            success_rate[1] += 1
            try:
                with open(filename) as training:
                    print('Reading file {0}...'.format(filename))
                    self.documents += 1
                    self.total_words.append(0)
                    for row in training:
                         #< separate row into sentences
                         row = row.strip().split('. ')
                         for sentence in row:
                            sentence = sentence.split()
                            #< format sentence
                            #< remove special things before/after word
                            formatted_sentence = [w.strip().lower() for w in sentence]
                            #< only add sentences longer than 0 words
                            if formatted_sentence:
                                self.vectorizer(formatted_sentence)
                                #< remove [''] and '' elements
                                while '' in formatted_sentence:
                                    formatted_sentence.remove('')
                                #< save sentence as a sequence of integers
                                self.superlist.append([self.vocabulary.index(x) for x in formatted_sentence]) #< emptied when data is SAVED
                                #< ??? self.superlist.append(list(map(lambda i: formatted_sentence.index(i)), formatted_sentence))

                    #< Print message to user
                    success_rate[0] += 1
                    print('Success!\n')

            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                continue

        return(success_rate)

    #< create weights and add index vectors to word vectors
    def apply_data(self, update = False):
        self.apply_weights(update)
        self.create_word_vectors(update)

    #< Assign weight values
    def apply_weights(self, update):
        print('Applying weights...\n')
        #< ??? Differrent weights if word is behind/after?
        #< Create weights for indexes
        for i, w in enumerate(self.weights):
            if update:
                self.word_vectors[i] = np.zeros(1024) #reset old word_vectors
            #< inverse document frequency + smoothing log(1 + (documents_with_term(t)/total_documents))
            #< document term frequency - freq(term, document_n)/words(document_n)
            idf = math.log1p(self.documents/len(self.weights[i][1:]))
            tf = [(e/self.total_words[n]) for n, e in enumerate(self.word_count[i]) if e != 0]
            self.weights[i][0] = sum(tf)*idf #tf-idf as tf * idf

    #< default update = False
    #< add index vectors in the words context to the word_vector
    def create_word_vectors(self, update):
        print('Creating word vectors...\n')

        #< sentece-generator
        gen_sentences = (x for x in self.superlist)

        #< read all sentences
        for sentence in gen_sentences:
            self.ngram_contexts(sentence)

        #< when updating with new data, re-do all previous vector additions
        if update:
            self.update_contexts()

    #< CBOW or skip-gram context. Specified at __init__
    #< Read context of word and add vectors
    def ngram_contexts(self, sentence):
        for i, word in enumerate(sentence):
            for n in range(1,self.window+1):
                #< words before
                if i - n >= 0: #< exclude negative indexes
                    try:
                        if self.context_type == 'CBOW':
                            prev_word = sentence[i-n]
                        elif self.context_type == 'skip-gram':
                            prev_word = sentence[i-n-1]

                        self.word_vectors[word] += (self.index_vectors[prev_word] * self.weights[prev_word][0])
                        self.evaled_data.append((word, prev_word))
                    except:
                        pass
                #< words after
                if i + n != len(sentence):
                    try:
                        if self.context_type == 'CBOW':
                            next_word = sentence[i+n]
                        elif self.context_type == 'skip-gram':
                            next_word = sentence[i+n+1]

                        self.word_vectors[word] += (self.index_vectors[next_word] * self.weights[next_word][0])
                        self.evaled_data.append((word, next_word))
                    except:
                        pass

    #< If update occurs w/o save, self.superlist contains the old data and it will be applied
    #< Updating of data
    def update_contexts(self):
        try:
            print('Re-applying updated vectors...\n')
            #< load previous vector additions
            data = list(np.load('/home/usr1/git/dist_data/d_data/{0}_hist.npy'.format(self.current_load)))

            #< redo additions with new weights
            for x, y in data:
                self.word_vectors[x] += (self.index_vectors[y] * self.weights[y][0])

            #< Save old data + new data
            data += self.evaled_data #evaled data contains all previous unsaved vector additions
            np.save('/home/usr1/git/dist_data/d_data/{0}_hist.npy'.format(self.current_load), data)

            #< discard data after save
            del data
            self.evaled_data = [] #< values used up!

        except FileNotFoundError as fnfe:
            if self.current_load == None:
                pass
            else:
                print(fnfe)
                pass #< when update is done with no saved file

    #######################################################
    ################## EVUALUATIONS #######################
    #######################################################

    def find_similarity(self, s_word1, s_word2):
        #< stem input
        word1, word2 = stem(s_word1), stem(s_word2)

        #< check if the words exists
        if word1 not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(s_word1)
        elif word2 not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(s_word2)
        else:
            i_word1 = self.word_vectors[self.vocabulary.index(word1)]
            i_word2 = self.word_vectors[self.vocabulary.index(word2)]

        cosine_sim = pw.cosine_similarity(i_word1.reshape(1,-1), i_word2.reshape(1,-1))

        return(cosine_sim[0][0])

    #< top 3
    def similarity_top(self ,s_word):
        #< stem input
        word = stem(s_word)

        if word not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(s_word)
        else:
            ind = self.vocabulary.index(word)
            word = self.word_vectors[ind]

        #< custom n for top
#        num = 6
#        top2 = [[0,""]]*num

        top = [[0, ""], [0, ""], [0, ""], [0, ""], [0, ""]]

        #< cosine sim between input word and ALL words
        for i, vect in enumerate(self.word_vectors):
            if i == ind:
                continue

            cs = pw.cosine_similarity(word.reshape(1,-1), vect.reshape(1,-1))

            #< Set highest values
            if cs > top[0][0]:
                top[0][0:] = cs, self.vocabulary[i]
            elif cs > top[1][0]:
                top[1][0:] = cs, self.vocabulary[i]
            elif cs > top[2][0]:
                top[2][0:] = cs, self.vocabulary[i]
            elif cs > top[3][0]:
                top[3][0:] = cs, self.vocabulary[i]
            elif cs > top[4][0]:
                top[4][0:] = cs, self.vocabulary[i]

        return(top)

    #< latent_semantic_analysis-like cosine similarity, takes data from HISTORY OF A SAVED FILE
    def lsasim(self, s_word1, s_word2):
        #< stem input
        word1, word2 = stem(s_word1), stem(s_word2)

        if word1 not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(word1)
        if word2 not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(word2)
        else:
            ind_1, ind_2 = self.vocabulary.index(word1), self.vocabulary.index(word2)

        data = np.load('/home/usr1/git/dist_data/d_data/{0}_hist.npy'.format(self.current_load))

        #< get all co-occurences of word1 and word2
        w1, w2 = [y for x, y in data if x == ind_1], [y for x, y in data if x == ind_2]
        del data

        #< create empty vectors
        vector_w1, vector_w2 = np.zeros(len(self.vocabulary)), np.zeros(len(self.vocabulary))

        #< populate vectors with co-occurences
        for x in w1:
            vector_w1[x] += 1 * self.weights[x][0]
        for x in w2:
            vector_w2[x] += 1 * self.weights[x][0]

        cosine_sim = pw.cosine_similarity(vector_w1.reshape(1, -1), vector_w2.reshape(1, -1))

        return cosine_sim[0][0]

    #< !!! ERRORS
    #< some form of graph, weight/freq atm
    def graph(self, word1, word2):

        if word1 != '' and word2 != '':
            word1, word2 = stem(word1), stem(word2)
            points_wf, points_sf = [], []
            points_wf.append(self.weights[self.vocabulary.index(word1)][0])
            points_wf.append(self.weights[self.vocabulary.index(word2)][0])
            points_sf.append(math.log(sum(self.word_count[self.vocabulary.index(word1)])))
            points_sf.append(math.log(sum(self.word_count[self.vocabulary.index(word2)])))
            
        else:
            points_wf = [x[0] for x in self.weights]
            points_sf = [sum(x) for x in self.word_count]

        plt.scatter(points_wf, points_sf)
        plt.xlabel("weight")
        plt.ylabel("log-e of freq")
        plt.show()


    #######################################################
    ################## DATA OPTIONS #######################
    #######################################################

    #< save data
    def save(self, filename):
        ofile = '/home/usr1/git/dist_data/d_data/{0}.npz'.format(filename)
        np.savez(ofile,
                 vocab = self.vocabulary,
                 w_vec = self.word_vectors,
                 i_vec = self.index_vectors,
                 weigh = self.weights,
                 wor_c = self.word_count,
                 t_wor = self.total_words,
                 docum = self.documents)

        histfile = '/home/usr1/git/dist_data/d_data/{0}_hist.npy'.format(filename)
        self.current_load = filename

        #< !!! Add some utility to the history
        #< save history
        if os.path.isfile(histfile):
            data = list(np.load(histfile))
            data += self.evaled_data
            np.save(histfile, data)
            self.evaled_data = []
        else:
            np.save(histfile, self.evaled_data)
            self.evaled_data = []

        self.superlist = [] #< Superlist (sentences from files) cleared from memory

    #< load saved data
    def load(self, filename):
        try:
            data = np.load('/home/usr1/git/dist_data/d_data/{0}.npz'.format(filename))

            self.current_load = filename

            self.vocabulary = list(data['vocab'])
            self.index_vectors = list(data['i_vec'])
            self.word_vectors = list(data['w_vec'])
            self.weights = list(data['weigh'])
            self.word_count = list(data['wor_c'])
            self.total_words = list(data['t_wor'])
            self.documents = int(data['docum'])

            #< clear data after data is extracted
            del data

        except FileNotFoundError as fnfe:
            return(fnfe)

    #< update the data
    def update(self, paths):
        files = []
        try:
            for path in paths:
                path = path.rstrip(',')
                files.append(path)

            status = self.process_data(files)
            print('{0}/{1} files successfully read'.format(status[0], status[1]))
            #< super strict atm, maybe aslong as status[0] > 1,
            # some data from files not successfully read might have been saved tho,
            # if error occured during reading
            if status[0] == status[1]:
                try:
                #< apply the new data with update=True
                    self.apply_data(True)
                    print('New data successfully applied')

                except Exception as e:
                    print('Error applying data\n{0}'.format(e))

        except Exception as e:
            print('Error reading data, try again\n{0}'.format(e))


    #< info about the data or individual words
    def info(self, arg_w):
        if arg_w != None:
            arg = stem(arg_w)

            if arg not in self.vocabulary:
                print(arg_w, 'does not exist')
            else:
                if arg_w != arg:
                    print('Word "{0}" stemmed to "{1}"'.format(arg_w, arg))
                print('\nFrequencies:')
                for i, c in enumerate(self.word_count[self.vocabulary.index(arg)]):
                    if c == 0:
                        continue
                    else:
                        print('Document {0}: {1}'.format(i+1, c))
                print('Total occurences: {0}\n'.format(sum(self.word_count[self.vocabulary.index(arg)])))
                print('Importance:')
                print('Term frequecy weight: {0}'.format(sum([(e/self.total_words[n]) for n, e in enumerate(self.word_count[self.vocabulary.index(arg)]) if e != 0])))
                print('Inverse document frequency: {0}'.format(math.log1p(self.documents/len(self.weights[self.vocabulary.index(arg)][1:]))))
                print('total weight of word: {0}\n'.format(self.weights[self.vocabulary.index(arg)][0]))
        else:
            print(len(self.vocabulary), 'unique words in vocabulary')
            print(sum(self.total_words), 'total words')
            print(self.documents, 'total documents')
            for i, c in enumerate(self.total_words):
                print('Document {0}: {1} words'.format(i+1, c))


#######################################################
##################### INTERFACE #######################
#######################################################

def main():

    distrib = DistributionalSemantics()

    print('Welcome to Distributial Semantics with Random Indexing\n')
    new_data = False
    #< init data
    while True:
        if new_data:
            print('Enter new data by typing "new <path>" finish by typing "apply"\n')
        else:
            print('Enter new data source by typing "new <path>" and load saved data by typing "load <name>"\n')

        setup = input('> ')
        setup = setup.split()

        if len(setup) == 0:
            print('Please try again')

        #< load saved data
        elif setup[0] == 'load':
            if not new_data:
                try:
                    distrib.load(setup[1])
                    break

                except Exception as e:
                    print('Try again\n', e)

        #< input a new data source
        elif setup[0] == 'new':
            new_data = True
#            status = distrib.process_data(['/home/usr1/git/dist_data/test_doc_1.txt'])
            status = distrib.process_data(['/home/usr1/git/dist_data/austen-emma.txt'])
            print('{0}/{1} files successfully read'.format(status[0], status[1]))
        #< apply precessed data
        elif setup[0] == 'apply':
            if new_data:
                distrib.apply_data()
                break
            else:
                print('Invalid command')
        #< exit
        elif setup[0] == 'exit':
            sys.exit()

        else:
            print('Invalid input')

    #< User interface after data has been loaded
    print('Type "sim <word1> <word2>" for similarity between two words, "top <word>" for top 3 similar words, "help" to display availible commands and "exit" to quit\n')
    while True:
        choice = input('> ')
        input_args = choice.split()

        #< empty input
        if not input_args:
            print('Please try again\n')

        #< RI similarity between words
        elif input_args[0] == 'sim':
            try:
                sim_res = distrib.find_similarity(input_args[1].lower(), input_args[2].lower())
                if sim_res == str(sim_res):
                    print(sim_res)
                else:
                    print('Cosine similarity between "{0}" and "{1}" is\n {2}\n'.format(input_args[1], input_args[2], sim_res))

            except Exception as e:
                print('Invalid input for "sim"\n', e)

        #< LSA similarity between words, [vocab*vocab] matrix
        elif input_args[0] == 'lsasim':
            try:
                lsa_res = distrib.lsasim(input_args[1], input_args[2])
                if lsa_res == str(lsa_res):
                    print(lsa_res)
                else:
                    print('LSA cosine similarity between "{0}" and "{1}" is\n {2}\n'.format(input_args[1], input_args[2], lsa_res))

            except Exception as e:
                print('Invalid input for "lsasim"\n', e)

        #< top 3 words
        elif input_args[0] == 'top':
            try:
                top_res = distrib.similarity_top(input_args[1].lower())
                if top_res == str(top_res):
                    print(top_res)
                else:
                    print('Top similar words for "{0}" is:'.format(input_args[1]))
                    for i, (dist, word) in enumerate(top_res):
                        print(i+1, dist[0][0], word)

            except Exception as e:
                print('Invalid input for "top"\n')

        #< quit
        elif input_args[0] == 'exit':
           break

        #< save data
        elif input_args[0] == 'save':
            try:
                distrib.save(input_args[1])
                print('Successfully saved as {0}\n'.format(input_args[1]))

            except Exception as e:
                print('Error\n{0}'.format(e))

        #< update data
        elif input_args[0] == 'update':
            try:
                distrib.update(input_args[1:])

            except Exception as e:
                print('Update failed\n{0}'.format(e))

        #< info about dataset or word
        elif input_args[0] == 'info':
            try:
                distrib.info(input_args[1].lower())

            except:
                distrib.info(None)

        #<
        elif input_args[0] == 'graph':
            try:
                distrib.graph(input_args[1], input_args[2])
            except:
                distrib.graph('','')

        #< help information
        elif input_args[0] == 'help':
            print('- Semantic operations')
            print('\t"sim <word> <word>" similarity between two words')
            print('\t"top <word>" top 3 similar words')
            print('\t"lsasim <word> <word>" LSA-like similarity between two words')
            print('- Data operations')
            print('\t"save <name>" save current data')
            print('\t"update <path>" update the data with a new textfile')
            print('\t"info" information about the data')
            print('\t"info <word>" information about a word')
            print('- ETC')
            print('\t"exit" to quit\n')

        else:
           print('Unrecognized command')

if __name__ == '__main__':
    main()


