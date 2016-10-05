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
            
            
            
    ######
        UPDATING DATA
       
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
        
        self.documents = 0
        self.weights = []
        self.total_words = 0
        self.sentences_total = 0
        
        self.gen_sentences = 0
        self.evaled_data = []
    
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
                
                if int(self.documents) in self.weights[self.vocabulary.index(word)][3:]:
                    pass
                else:
                    self.weights[self.vocabulary.index(word)].append(int(self.documents))
                
                if word not in added:
                    self.weights[self.vocabulary.index(word)][2] += 1
                    added.append(word)   
                
    def create_vectors(self, filenames):
        err = []
        for t, filename in enumerate(filenames):
            try:
                with open(filename) as training:
                    print("Reading file", filename, "...")
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
                    print("Success!\n")
                    
            except FileNotFoundError as fnfe:
                print('FILE ERROR!\n {0}\n'.format(fnfe))
                err.append(filename)
                continue
        
        #generator for sentences
        self.gen_sentences = (x for x in self.superlist) 
        
        #return values
        if err:
            return err
        else:
            return None
        
    #create weights and add index vectors to word vectors
    def apply(self):
        self.apply_weights()
        self.create_word_vectors()
                                
    ######## weights always updated after create_vectors, incorporate       
    # n-gram history? could be saved 
                                
    #Assign weight values
    def apply_weights(self):  
        print("Applying weights...\n")
        ###### Differrent weights if word is behind/after???
        ##Create weights for indexes
        for i, w in enumerate(self.weights):
#            self.word_vectors[i] = np.zeros(1024) #updating
            #real idf
            idf = math.log1p(self.documents/len(self.weights[i][3:])) #smooth
            isf = math.log(self.sentences_total/w[2]) #inverse sentence frequency
            self.weights[i][0] = (w[1]/self.total_words)*idf
            
    #add index vectors in the words context to the word_vector
    def create_word_vectors(self):
        print("Creating word vectors...\n")
        window = 1 #how many words before/after to consider being a part of the context
        #weight for sliding window > 1
        #create contexts
        for sentence in self.gen_sentences:
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
#        self.superlist = []
           
    
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
    #save data
    def save(self):
        ofile = '/home/usr1/git/dist_data/np_data.npz'
        np.savez(ofile, 
                 v = self.vocabulary, 
                 wv = self.word_vectors, 
                 iv = self.i_vectors, 
                 w = self.weights, 
                 tw = self.total_words, 
                 st = self.sentences_total, 
                 d = self.documents)    
                 
    #load saved data
    def load(self):
        data = np.load('/home/usr1/git/dist_data/np_data.npz') 
        
        self.vocabulary = list(data['v'])
        self.i_vectors = list(data['iv'])
        self.word_vectors = list(data['wv'])   
        self.weights = list(data['w'])
        self.total_words = int(data['tw'])
        self.sentences_total = int(data['st'])
        self.documents = int(data['d'])
    
    #update the data
    def update(self, paths):
        #read new files
        for path in paths:
            path.rstrip(',')
            z = self.create_vectors([path])
            
            ### HANDLE ERRORS
        if z != None:
            print("Error reading", z)            
        else:
            #apply the new data
            self.apply()
            print("New data successfully applied")
        
    #info about the data or individual words
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
    
    x = distrib_semantics()
    
    print("Welcome to Distributial Semantics with Random Indexing")
    new_data = False
    #load data
    while True:
        if new_data == True:
            print("Enter paths by typing 'new <path>' finish by typing 'apply'")
        else:
            print("Enter new data source by typing 'new <path>' and load saved data by typing 'load'")
            
        setup = input('> ')
        setup = setup.split()
        
        if setup[0] == 'load':
            if new_data == False:
                x.load()
                break
        
        elif setup[0] == 'new':
            new_data = True
            x.create_vectors(['/home/usr1/PythonPrg/project/test_doc_1.txt','/home/usr1/PythonPrg/pr111oject/test_doc_2.txt', '/home/usr1/PythonPrg/project/test_doc_3.txt', '/home/usr1/PythonPrg/project/test_doc_4.txt'])
                
        elif setup[0] == 'apply':
            if new_data == True:
                x.apply()
                break

        else:
            print('Invalid input')
        
    #User interface after data has loaded
    print("Type 'sim <word1> <word2>' for similarity between two words, 'top <word>' for top 3 similar words, 'help' to display availible commands and 'exit' to quit")
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
                    print('Cosine similarity between', input_args[1], 'and',input_args[2] ,'is\n', sim_res)
            except:
                print("Invalid input for 'sim'")
        
        #DO NOT USE, CONTAINS ERRORS        
        #fix error output
        elif input_args[0] == 'top':
            try:
                top_res = x.similarity_top(input_args[1].lower())
                if top_res == str(top_res):
                    print(top_res)
                else:
                    print('Top similar words for', input_args[1], 'is:')
                    for i, (x, y) in enumerate(top_res):
                        print(i+1, x, y)
            except:
                print("Invalid input for 'top'")
               
        elif input_args[0] == 'exit':
           break
       
        elif input_args[0] == 'save':
            x.save()
        
        elif input_args[0] == 'update':
            try:
                x.update(input_args[1:])
            except:
                print('Please provide a path/paths')
            
        elif input_args[0] == 'info':
            try:
                x.info(input_args[1].lower())
            except:
                x.info(None)
                
        elif input_args[0] == 'graph':
            x.graph()
            
        elif input_args[0] == 'help':
            print("- Semantic operations")
            print("'sim <word1> <word2>' for similarity")
            print("'top <word>' for top 3 similar words")     
            print("- Data operations")
            print("'save' to save current data")
            print("'update <path>' to update the data with a new textfile")
            print("'info' for info about the data")
            print("'info <word> for info about the word'")
            print("- ETC")
            print("'exit' to quit")
        
        else:
           print("Unrecognized command")
    
if __name__ == '__main__':
    main()


