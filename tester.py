# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:51:47 2016
@author: Adam Ek
"""
from WSM import DataReader
from WSM import RandomVectorizer
from WSM import TermRelevance
from WSM import Contexter
from WSM import Similarity
from WSM import DataOptions
import tsne
import matplotlib.pyplot as plt
import numpy as np


def main():
    plotting = False
#    load_oldsave()
    new_save(plotting)

def load_oldsave():
    do = DataOptions()
    vectors, documents, data_info = do.load('tester')
    sim = Similarity(vectors)

    words1 = ['the', 'when', 'speak', 'run', 'high', 'flow', 'love', 'he']
    words2 = ['and', 'where', 'talk', 'walk', 'low', 'water', 'hate', 'she']
    for w1, w2 in zip(words1, words2):
        print(w1, w2)
        s = sim.cosine_similarity(w1, w2)
        print(s)

def new_save(plting):
    plotting = plting

#    dataset1 = ['/home/usr1/git/dist_data/austen-persuasion.txt']
#    dataset1 = ['/home/usr1/git/dist_data/austen-emma.txt', '/home/usr1/git/dist_data/austen-persuasion.txt', '/home/usr1/git/dist_data/austen-sense.txt']
    dataset1 = ['/home/usr1/git/dist_data/test_doc_3.txt']
#    dataset1 = ['/home/usr1/git/dist_data/formatted1.txt']

#DATAREADER
##################################################
    dr = DataReader()
    #read the file dataset1 and output all sentences, all words, and information about word count/documents
    sentences, vocabulary, documents = dr.preprocess_data(dataset1)

#    for sent in sentences:
#        print(sent, '\n')
#    print(vocabulary)
#    print(documents)

    print('reading file done')
##################################################

##RANDOMVECTORIZER
###################################################
    rv = RandomVectorizer(dimensions=512)
#    #create word and random vectors for the strings in vocabulary
    vectors = rv.vocabulary_vectorizer(vocabulary)
    print('vectoring done')
#    print(vectors)
###################################################

##WEIGHTER
###################################################
#    #init Weighter, with scheme 0 and don't do idf
    tr = TermRelevance(documents, scheme=2, doidf=False, smooth_idf=False)

    #weight the dictionary of vectors
    vectors = tr.weight(vectors)

##################################################
#
##CONTEXTER
##################################################
#    #Init Contexter
    t, s = 0, 2
    cont1 = Contexter(vectors, contexttype=t, window=1, context_scope=s)
    cont3 = Contexter(vectors, contexttype=t, window=3, context_scope=s)
    cont5 = Contexter(vectors, contexttype=t, window=5, context_scope=s)
    cont10 = Contexter(vectors, contexttype=t, window=10, context_scope=s)

    vector_vocabulary1 = cont1.process_data(sentences)
    vector_vocabulary3 = cont3.process_data(sentences)
    vector_vocabulary5 = cont5.process_data(sentences)
    vector_vocabulary10 = cont10.process_data(sentences)
    print('read_context done')
#
    ###PPMI of co-occurence
#    ppmi, vocab = cont1.PPMImatrix(cont1.vocabt)
#    print(vocab)
#    for i, m in enumerate(ppmi):
#        print(vocab[i])
#        print(m, '\n')
#    print(cc)

    print('reading contexts done')
###################################################
#
##DATAOPTIONS
###################################################
#    # initialize DataOptions for saving/extracting information
#    do = DataOptions()
#    # save the data
#    do.save('tester', vector_vocabulary5, documents, cont5.data_info, wgt.weight_setup)
###################################################

##SIMILARITY
###################################################
#    wc = 0
#    for x in documents:
#        for v in documents[x].values():
#            wc += v
#    print(wc)
##    #Initialize similarity class
#    sim1 = Similarity(vector_vocabulary1)
#    sim3 = Similarity(vector_vocabulary3)
#    sim5 = Similarity(vector_vocabulary5)
#    sim10 = Similarity(vector_vocabulary10)
##
###    words1 = ['the', 'when', 'speak', 'run', 'high', 'flow', 'love', 'he']
###    words2 = ['and', 'where', 'talk', 'walk', 'low', 'water', 'hate', 'she']
#    words1 = ['language', 'applied']
#    words2 = ['linguist', 'theoretical']
#    for w1, w2 in zip(words1, words2):
#        print(w1, w2)
#        s1 = sim1.cosine_similarity(w1, w2)
#        s3 = sim3.cosine_similarity(w1, w2)
#        s5 = sim5.cosine_similarity(w1, w2)
#        s10 = sim10.cosine_similarity(w1, w2)
#        print(s1)
#        print(s3)
#        print(s5)
#        print(s10)
#        print((s1+s3+s5+s10)/4)

#################################################

#PLOTTING
##################################################
    if plotting:
        ar = []
        lbs = []
        for i, v in enumerate(vector_vocabulary5):
            if i%5 == 0:
                ar.append(vector_vocabulary5[v])
                lbs.append(v)

#        lbs = ['the', 'when', 'speak', 'run', 'high', 'flow', 'love', 'he', 'and', 'where', 'talk', 'walk', 'low', 'water', 'hate', 'she']
#        for word in lbs:
#            ar.append(vector_vocabulary5[word])

        arrs = np.array(ar)
        Y = tsne.tsne(arrs, 2, 50, 20.0)
#        print(Y)
        fig, ax = plt.subplots()
        ax.scatter(Y[:,0], Y[:,1], 20)

        for i, name in enumerate(lbs):
            ax.annotate(name, (Y[i][0], Y[i][1]))

        plt.show()

if __name__ == '__main__':
    main()