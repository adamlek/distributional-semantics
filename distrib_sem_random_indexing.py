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

INVESTIGATE
    update function:
        self.evaled_data      
        
TO ADD:
    * lemma/stemming
    * shared information [from history]
       
"""
import numpy as np
import re
import random
import math
import sklearn.metrics.pairwise as d
import matplotlib.pyplot as plt
import os.path

class distrib_semantics():
    
    def __init__(self):
        self.vocabulary = []
        self.index_vectors = []
        self.word_vectors = []
        self.superlist = []
        
        self.documents = 0
        self.weights = []
        self.total_words = 0
        self.sentences_total = 0
        
        self.evaled_data = [] # !!! check values, etc.
        #< Current instance, changed when loading/saving
        self.current_load = None
    
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
        added = [] #< Unique words in sentence
        self.sentences_total += 1
        for i, word in enumerate(formatted_sentence):
                    
            #< remove special things inside words
            formatted_sentence[i] = re.sub('[^a-zåäö0-9%-]', '', formatted_sentence[i])    
            word = re.sub('[^a-zåäö0-9%-]', '', word)

            #< dont add null words
            if word == '':
                continue 
                
            self.total_words += 1
            
            #< FINE TUNE DATA 
            #< change numbers to NUM
            if word.lstrip('-').isdigit():
                word = 'NUM'
                formatted_sentence[i] = 'NUM'
            #< NUMS -> '5-6' etc
            elif re.match('\d+-\d+', word) is not None:
                word = 'NUMS'
                formatted_sentence[i] = 'NUMS'
            #< percentages -> 12% etc
            elif '%' in word:
                word = 'PERC'
                formatted_sentence[i] = 'PERC'
            
            #< set up vectors
            if word not in self.vocabulary:
                self.vocabulary.append(word)
                self.index_vectors.append(self.rand_index_vector())
                self.word_vectors.append(np.zeros(1024))
                self.weights.append([0, 1, 1, self.documents])
            #< create weight tools
            else:
                self.weights[self.vocabulary.index(word)][1] += 1
                
                if int(self.documents) in self.weights[self.vocabulary.index(word)][3:]:
                    pass
                else:
                    self.weights[self.vocabulary.index(word)].append(int(self.documents))
                #< unique words in sentence
                if word not in added:
                    self.weights[self.vocabulary.index(word)][2] += 1
                    added.append(word)
                
    def create_vectors(self, filenames):
        for filename in filenames:
            try:
                with open(filename) as training:
                    print('Reading file {0}...'.format(filename))
                    self.documents += 1
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
                                self.superlist.append([self.vocabulary.index(x) for x in formatted_sentence])
                                # ??? self.superlist.append(list(map(lambda i: formatted_sentence.index(i)), formatted_sentence))
                      
                    #< Print message to user
                    print('Success!\n')
                    
            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                continue    

    #< create weights and add index vectors to word vectors
    def apply(self, update = False):
        self.apply_weights(update)
        self.create_word_vectors(update)

    #< Assign weight values
    def apply_weights(self, update):
        print('Applying weights...\n')
        # !!! Differrent weights if word is behind/after?
        #< Create weights for indexes
        for i, w in enumerate(self.weights):
            if update:
                self.word_vectors[i] = np.zeros(1024) #reset old word_vectors
            #< inverse document frequency + smoothing
            idf = math.log1p(self.documents/len(self.weights[i][3:])) #smooth
#            isf = math.log(self.sentences_total/w[2]) #inverse sentence frequency
            self.weights[i][0] = (w[1]/self.total_words)*idf
            
    #< default update = False
    #< add index vectors in the words context to the word_vector
    def create_word_vectors(self, update):
        print('Creating word vectors...\n')
        self.window = 1 #< how many words before/after to consider being a part of the context

        gen_sentences = (x for x in self.superlist)
        
        for sentence in gen_sentences:
            self.ngram_contexts(sentence)
        
        #< create contexts
        if update:
            self.update_contexts()
                
    #< Read context of word and add vectors    
    def ngram_contexts(self, sentence):
        for i, word in enumerate(sentence):
            for n in range(1,self.window+1):
                #< words before
                try:
                    if i - n >= 0: #< exclude negative indexes
                        prev_word = sentence[i-n]
                        self.word_vectors[word] += (self.index_vectors[prev_word] * self.weights[prev_word][0])
                        self.evaled_data.append((word, prev_word))
                except: #< no word before
                    pass
                
                #< words after
                try:
                    next_word = sentence[i+n]
                    self.word_vectors[word] += (self.index_vectors[next_word] * self.weights[next_word][0])
                    self.evaled_data.append((word, next_word))
                except: #< no word after
                    pass
    
    #< Updating of data
    def update_contexts(self):
        try:
            print('Re-applying updated vectors...\n')
            #< load previous vector addition
            data = list(np.load('/home/usr1/git/dist_data/d_data/{0}_hist.npy'.format(self.current_load)))
            
            #< redo additions with new weights
            for x, y in data:
                print(x, y)
                self.word_vectors[x] += (self.index_vectors[y] * self.weights[y][0])
            
            #< Save old data + new data 
            data += self.evaled_data
            np.save('/home/usr1/git/dist_data/d_data/{0}_hist.npy'.format(self.current_load), data)
            
            #< discard data after save
            del data
            self.evaled_data = []#< values used up!
        
        except FileNotFoundError as fnfe:
            print(fnfe)
            pass #< when update is done with no save
        
    #######################################################
    ################## EVUALUATIONS #######################
    #######################################################
        
    def find_similarity(self, word1, word2):
        #< check if the words exists
        if word1 not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(word1)
        elif word2 not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(word2)
        else:
            i_word1 = self.word_vectors[self.vocabulary.index(word1)]
            i_word2 = self.word_vectors[self.vocabulary.index(word2)]
            
        #< Cosine similarity
        cosine_sim = d.cosine_similarity(i_word1.reshape(1,-1), i_word2.reshape(1,-1))
        
        return(cosine_sim[0][0])        

    #< top 3
    def similarity_top(self, word):
        if word not in self.vocabulary:
            return '{0} does not exist, try again\n'.format(word)
        else:
            ind = self.vocabulary.index(word)
            word = self.word_vectors[ind]
        
        top = [[0,""],[0,""],[0,""]]
        
        #< cosine sim between input word and ALL words
        for i, vect in enumerate(self.word_vectors):
            if i == ind:
                continue
            
            cs = d.cosine_similarity(word.reshape(1,-1), vect.reshape(1,-1))
            
            #< Set highest values
            if cs > top[0][0]:
                top[0][0:] = cs, self.vocabulary[i]
            elif cs > top[1][0]:
                top[1][0:] = cs, self.vocabulary[i]
            elif cs > top[2][0]:
                top[2][0:] = cs, self.vocabulary[i]
                
        return(top)
    
    #< some form of graph
    def graph(self):
        
        points_wf = [x[1] for x in self.weights]
        points_sf = [math.log(x[2]) for x in self.weights]
        
        plt.scatter(points_wf, points_sf)
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
                 t_wor = self.total_words, 
                 sen_t = self.sentences_total, 
                 docum = self.documents)
        
        histfile = '/home/usr1/git/dist_data/d_data/{0}_hist.npy'.format(filename)
        self.current_load = filename
        
        #< !!! Add some utility to the history
        #< if the file exists, its updated, don't save
        if os.path.isfile(histfile):
            pass
        else:
            np.save(histfile, self.evaled_data)
            self.evaled_data = []
        
        self.superlist = [] #< Superlist (sentences from files) cleared from memory
                 
    #< load saved data
    def load(self, filename):
        data = np.load('/home/usr1/git/dist_data/d_data/{0}.npz'.format(filename))
        
        self.current_load = filename
        
        self.vocabulary = list(data['vocab'])
        self.index_vectors = list(data['i_vec'])
        self.word_vectors = list(data['w_vec'])
        self.weights = list(data['weigh'])
        
        self.total_words = int(data['t_wor'])
        self.sentences_total = int(data['sen_t'])
        self.documents = int(data['docum'])
        
        #< clear data after data is extracted
        del data
    
    #< update the data
    def update(self, paths):
        try:
            for path in paths:
                path.rstrip(',')
                self.create_vectors([path])
            
            #< apply the new data with update=True
            self.apply(True)
            print('New data successfully applied')
            
        except Exception as e:
            print(e)

    #< info about the data or individual words
    def info(self, arg):
        if arg != None:
            if arg not in self.vocabulary:
                print(arg, 'does not exist')
            else:
                print('occurences:', self.weights[self.vocabulary.index(arg)][1])
                print('in', self.weights[self.vocabulary.index(arg)][2], 'sentences')
                print('word in documents:', self.weights[self.vocabulary.index(arg)][3:])
                print('tf:', self.weights[self.vocabulary.index(arg)][1]/self.total_words)
                print('idf:', math.log1p(self.documents/len(self.weights[self.vocabulary.index(arg)][3:])))
                print('total weight of word:', self.weights[self.vocabulary.index(arg)][0],'\n')
        else:
            print(len(self.vocabulary), 'unique words in vocabulary')
            print(self.sentences_total, 'total sentences')
            print(self.total_words, 'total words')
            print(self.documents, 'total documents\n')
            
#######################################################
##################### INTERFACE #######################
#######################################################
def main():
    
    distrib = distrib_semantics()
    
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
                    print('Try again\n{0}'.format(e))
        
        #< input a new data source
        elif setup[0] == 'new':
            new_data = True
#            distrib.create_vectors(['/home/usr1/git/dist_data/test_doc_3.txt'])
            distrib.create_vectors(['/home/usr1/git/dist_data/austen-emma.txt'])
                
        #< apply precossed data
        elif setup[0] == 'apply':
            if new_data:
                distrib.apply()
                break
            else:
                print('Invalid command')
        #< exit
        elif setup[0] == 'exit':
            break

        else:
            print('Invalid input')
        
    #< User interface after data has been loaded
    print('Type "sim <word1> <word2>" for similarity between two words, "top <word>" for top 3 similar words, "help" to display availible commands and "exit" to quit\n')
    while True:
        choice = input('> ')  
        input_args = choice.split()
        
        #< empty input
        if not input_args:
            print('Please try again')
            
        #< similarity between words
        elif input_args[0] == 'sim':
            try:
                sim_res = distrib.find_similarity(input_args[1].lower(), input_args[2].lower())
                if sim_res == str(sim_res):
                    print(sim_res)
                else:
                    print('Cosine similarity between', input_args[1], 'and', input_args[2], 'is\n', sim_res, '\n')
            
            except Exception as e:
                print(e)
                print('Invalid input for "sim"')
        
        #< top 3 words
        elif input_args[0] == 'top':
            try:
                top_res = distrib.similarity_top(input_args[1].lower())
                if top_res == str(top_res):
                    print(top_res)
                else:
                    print('Top similar words for "{0}" is:'.format(input_args[1]))
                    for i, (dist, word) in enumerate(top_res):
                        print(i+1, dist, word)
                        
            except Exception as e:
                print('Invalid input for "top"\n{0}'.format(e))
        
        #< quit
        elif input_args[0] == 'exit':
           break
       
        #< save data
        elif input_args[0] == 'save':
            try:
                x.save(input_args[1])
                print('Successfully saved as', input_args[1])
                
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
            distrib.graph()
        
        #< help information
        elif input_args[0] == 'help':
            print('- Semantic operations')
            print('\t"sim <word1> <word2>" for similarity')
            print('\t"top <word>" for top 3 similar words')
            print('- Data operations')
            print('\t"save <name>" to save current data')
            print('\t"update <path>" to update the data with a new textfile')
            print('\t"info" for info about the data')
            print('\t"info <word>" for info about the word')
            print('- ETC')
            print('\t"exit" to quit')
        
        else:
           print('Unrecognized command')
    
if __name__ == '__main__':
    main()


