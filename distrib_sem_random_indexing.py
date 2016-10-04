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
            
    Init
        summ of vectors
        normalize vec
       
"""
import numpy as np
import re
import random
import math
import sklearn.metrics.pairwise as d
import matplotlib.pyplot as plt

class distrib_semantics():
    
    def __init__(self):
        self.vocabulary = []
        self.i_vectors = []
        self.word_vectors = []   
        self.superlist = []
        self.svded = []
        
        self.documents = 0
        self.weights = []
        self.total_words = 0
        self.sentences_total = 0
    
    #create an index vector for each word
    def rand_index_vector(self):
        arr = np.zeros(1024)        

        for i in range(0, 4):
            if i%2 == 0:
                arr[random.randint(0, 1023)] = 1
            else:
                arr[random.randint(0, 1023)] = -1
        
        return arr
        
    def vectorizer(self, formatted_sentence):
        added = []
        self.sentences_total += 1
        for i, word in enumerate(formatted_sentence):
                    
            #remove special things inside words
            formatted_sentence[i] = re.sub('[^a-zåäö0-9%-]', '', formatted_sentence[i])    
            word = re.sub('[^a-zåäö0-9%-]', '', word)

            #len(word) == 0:
            #dont add null words
            if word == '':
                continue #continue does not add ''
                
            self.total_words += 1    
            ###### FINE TUNE DATA
            #change numbers to NUM
            if word.lstrip('-').isdigit():
                word = 'NUM'
                formatted_sentence[i] = 'NUM'
            ### NUMS -> '5-6' etc
            elif re.match('\d+-\d+', word) is not None:
                word = 'NUMS'
                formatted_sentence[i] = 'NUMS'
            #percentages -> 12% etc
            elif '%' in word:
                word = 'PERC'
                formatted_sentence[i] = 'PERC'
            
            #set up vectors
            if word not in self.vocabulary:
                self.vocabulary.append(word)
                self.i_vectors.append(self.rand_index_vector())
                self.word_vectors.append(np.zeros(1024))
                self.weights.append([0, 1, 1, self.documents])        
            #create weight tools
            else:
                self.weights[self.vocabulary.index(word)][1] += 1
                
                if self.documents in self.weights[self.vocabulary.index(word)][3:]:
                    pass
                else:
                    self.weights[self.vocabulary.index(word)].append(self.documents)
                
                if word not in added:
                    self.weights[self.vocabulary.index(word)][2] += 1
                    added.append(word)   
                
    def create_vectors(self, filenames):
        for filename in filenames:
            try:
                with open(filename) as training:
                    self.documents += 1
                    for row in training:
                         #separate row into sentences
                         row = row.strip().split('. ')
                         for sentence in row:
                            sentence = sentence.split()
                            #format sentence
                            #remove special things before/after word
                            formatted_sentence = [w.strip().lower() for w in sentence]
                            #only add sentences longer than 0 words    
                            if len(formatted_sentence) > 0:
                                #create index_vector for word
                                #currently takes a senttence, change so it takes a word???
                                self.vectorizer(formatted_sentence)
                                
                                #remove [''] and '' elements
                                while '' in formatted_sentence:
                                    formatted_sentence.remove('')
                                
                                self.superlist.append(formatted_sentence)
            
            except FileNotFoundError:
                print("File not found, try again")
                
        
        self.apply_weights()
        self.create_word_vectors()
                                
    def apply_weights(self):        
        ###### Differrent weights if word is behind/after???
        ##Create weights for indexes
        for i, w in enumerate(self.weights):
            #tf-idf weight
            #real idf
            idf1 = math.log(self.documents/len(self.weights[i][3:]))
            idf = math.log(self.sentences_total/w[2])
            self.weights[i][0] = (w[1]/self.total_words)*idf
            
    def create_word_vectors(self):
        window = 1 #how many words before/after to consider being a part of the context
        #weight for sliding window > 1
        #create contexts
        for sentence in self.superlist:
            for i, word in enumerate(sentence):
                ###SLIDING WINDOW PARAMETER                                
                for n in range(1,window+1):
                    #words before                                
                    try:
                        if i - n >= 0: #exclude negative indexes
                            prev_word = self.vocabulary.index(sentence[i-n]) 
                            self.word_vectors[self.vocabulary.index(word)] += (self.i_vectors[prev_word] * self.weights[prev_word][0])
                    except:
                        pass
                    
                    #words after
                    try:
                        next_word = self.vocabulary.index(sentence[i+n])
                        self.word_vectors[self.vocabulary.index(word)] += (self.i_vectors[next_word] * self.weights[next_word][0])
                    except:
                        pass
        #empty sentence list incase update command
        self.superlist = []
           
    
    #######################################################
    ################## EVUALUATIONS #######################
    #######################################################
    def find_similarity(self, word1, word2):
        
        #check if the words exists
        if word1 not in self.vocabulary:
            return ' '.join([word1, "does not exist, try again"])
        elif word2 not in self.vocabulary:
            return ' '.join([word2, "does not exist, try again"])
        else:
            i_word1 = self.word_vectors[self.vocabulary.index(word1)]
            i_word2 = self.word_vectors[self.vocabulary.index(word2)]
            
        ### Check definition and implementation
        cosine_sim = d.cosine_similarity(i_word1.reshape(1,-1), i_word2.reshape(1,-1))
        
        return(cosine_sim[0][0])        

    #top 3
    def similarity_top(self, word):
        
        if word not in self.vocabulary:
            return ' '.join([word, "does not exist, try again"])
        else:        
            ind = self.vocabulary.index(word)
            word = self.word_vectors[ind]
        
        top = [0,0,0] 
        word_top = ["","",""]
        
        for i, vect in enumerate(self.word_vectors):
            if i == ind:
                continue
            
            cs = d.cosine_similarity(word.reshape(1,-1), vect.reshape(1,-1))
            
            if cs > top[0]:
                top[0] = cs
                word_top[0] = self.vocabulary[i]
            elif cs > top[1]:
                top[1] = cs
                word_top[1] = self.vocabulary[i]
            elif cs > top[2]:
                top[2] = cs
                word_top[2] = self.vocabulary[i]
                
        return [(x, y[0][0]) for x, y in zip(word_top,top)]
    
    def graph(self):
        
        points = []        
        
        for val in self.weights:
            points.append(val[1])
        
        plt.scatter(points, points)
        plt.show()
            
        
    #######################################################
    ################## DATA OPTIONS #######################
    #######################################################
    def save(self):
        outfile1 = '/home/usr1/git/dist_data/np_1.npy'
        outfile2 = '/home/usr1/git/dist_data/np_2.npy'
        outfile3 = '/home/usr1/git/dist_data/np_3.npy'
        outfile4 = '/home/usr1/git/dist_data/np_4.npy'
        outfile5 = '/home/usr1/git/dist_data/np_5.npy'
        outfile6 = '/home/usr1/git/dist_data/np_6.npy'
        outfile7 = '/home/usr1/git/dist_data/np_7.npy'
        
        np.save(outfile1, self.vocabulary)
        np.save(outfile2, self.word_vectors)
        np.save(outfile3, self.i_vectors)
        np.save(outfile4, self.weights)
        np.save(outfile5, self.total_words)
        np.save(outfile6, self.sentences_total)
        np.save(outfile7, self.documents)
    
    def load(self):
        self.vocabulary = list(np.load('/home/usr1/git/dist_data/np_1.npy'))
        self.word_vectors = list(np.load('/home/usr1/git/dist_data/np_2.npy'))
        self.i_vectors = list(np.load('/home/usr1/git/dist_data/np_3.npy'))
        self.weights = list(np.load('/home/usr1/git/dist_data/np_4.npy'))
        self.total_words = np.load('/home/usr1/git/dist_data/np_5.npy')
        self.sentences_total = np.load('/home/usr1/git/dist_data/np_6.npy')
        self.documents = np.load('/home/usr1/git/dist_data/np_7.npy')
    
    def update(self, path):
        self.create_vectors([path])
        
    def info(self, arg):

        if arg != None:
            if arg not in self.vocabulary:
                print(arg, 'does not exist')
            else:
                print(arg)
                print('occurences:', self.weights[self.vocabulary.index(arg)][1])
                print('in', self.weights[self.vocabulary.index(arg)][2], 'sentences')
                print('total weight of word:', self.weights[self.vocabulary.index(arg)][0])
                print('word in documents:', self.weights[self.vocabulary.index(arg)][3:])
                print('idf', math.log(self.documents/len(self.weights[self.vocabulary.index(arg)][3:])))

        else:

            print(len(self.vocabulary), 'unique words in vocabulary')
            print(self.sentences_total, 'total sentences')
            print(self.total_words, 'total words')
            print(self.documents, 'total documents')
            
    
def main():
#    x = distrib_semantics(['/home/usr1/Python_Prg_1/SU_PY/project/gutenberg/austen-emma.txt'])    
    x = distrib_semantics()

    #,'/home/usr1/PythonPrg/project/gutenberg/austen-emma.txt','/home/usr1/PythonPrg/project/gutenberg/austen-sense.txt'
    
    print("Welcome to Distributial Semantics with Random Indexing")
    
    #load data
    while True:
        print("Enter new data source by typing 'new path', load by typing 'load'")
        setup = input('> ')
        setup = setup.split()
        if setup[0] == 'load':
            x.load()
            break
        elif setup[0] == 'new':
            x.create_vectors(['/home/usr1/PythonPrg/project/test_doc_1.txt'])
            break
        else:
            print('Invalid input')
        
    #User interface after data has loaded
    print("Type 'sim word1 word2' for similarity between two words, 'top word' for top 3 similar words and 'exit' to quit")
    while True:
        choice = input('> ')  
        input_args = choice.split()
        
        if len(input_args) == 0:
            print('Please try again')
        #fix error output
        elif input_args[0] == 'sim':
            try:
                sim_res = x.find_similarity(input_args[1].lower(), input_args[2].lower())
                if sim_res == str(sim_res):
                    print(sim_res)
                else:
                    print('Cosine similarity between', input_args[1], 'and',input_args[2] ,'is\n', sim_res, '\n')
            except:
                print("Invalid input for 'sim'")
        
        #DO NOT USE, CONTAINS ERRORS        
        #fix error output
        elif input_args[0] == 'top':
            try:
                top_res = x.similarity_top(input_args[1].lower())
                print('Top similar words for', input_args[1], 'is:')
                for i, (x, y) in enumerate(top_res):
                    print(i+1, x, y)
            except:
                print("Invalid input for top'")
               
        elif input_args[0] == 'exit':
           break
       
        elif input_args[0] == 'save':
            x.save()
        
        elif input_args[0] == 'update':
            try:
                x.update(input_args[1])
            except:
                print('Please provide a path')
            
        elif input_args[0] == 'info':
            try:
                x.info(input_args[1].lower())
            except:
                x.info(None)
                
        elif input_args[0] == 'graph':
            x.graph()
            
        elif input_args[0] == 'help':
            print("sim word1 word2' for similarity")
            print("'top word' for top 3 similar words")        
            print("'save' to save current data")
            print("'update path' to update the data with a new textfile")
            print("'info' for info about the data")
            print("'exit' to quit")
        
        else:
           print("Unrecognized command")
    
if __name__ == '__main__':
    main()


