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
import csv
import scipy.stats as st


def main():
#    load_oldsave()
    new_save()

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

def new_save():
    plotting = False
    teststuff = True

    dataset1 = ['/home/usr1/git/dist_data/test_doc_1.txt', '/home/usr1/git/dist_data/test_doc_2.txt', '/home/usr1/git/dist_data/test_doc_3.txt', '/home/usr1/git/dist_data/test_doc_4.txt', '/home/usr1/git/dist_data/test_doc_5.txt', '/home/usr1/git/dist_data/test_doc_6.txt']
#    dataset1 = ['/home/usr1/git/dist_data/austen-emma.txt', '/home/usr1/git/dist_data/austen-persuasion.txt', '/home/usr1/git/dist_data/austen-sense.txt']
#    dataset1 = ['/home/usr1/git/dist_data/reut1.txt']
#    dataset1 = ['/home/usr1/git/dist_data/formatted2.txt', '/home/usr1/git/dist_data/reut1.txt', '/home/usr1/git/dist_data/reut2.txt']

#DATAREADER
##################################################
    dr = DataReader()
    #read the file dataset1 and output all sentences, all words, and information about word count/documents
    sentences, vocabulary, documents = dr.preprocess_data(dataset1)

#    print(len(sentences))
#    for sent in sentences:
#        print(sent, '\n')

    t = 0
    for v in documents:
        t += sum(documents[v].values())
    print(t, 'total tokens')
    print(len(vocabulary), 'total types')

    print('reading file done')
##################################################

##RANDOMVECTORIZER
###################################################
    rv = RandomVectorizer(dimensions=1024, random_elements=4)
    #create word and random vectors for the strings in vocabulary
    vectors = rv.vocabulary_vectorizer(vocabulary)
    print('vectoring done')
###################################################

##WEIGHTER
###################################################
    w, t, s = 1, 0, 2
    print(w, t, s)
    #init Weighter, with scheme 0 and don't do idf
    tr = TermRelevance(documents, scheme=w, doidf=True, smooth_idf=True)

    #weight the dictionary of vectors
    vectors = tr.weight(vectors)

##################################################
#
##CONTEXTER
##################################################
    #Init Contexter
    cont1 = Contexter(vectors, contexttype=t, window=1, context_scope=s)
    cont5 = Contexter(vectors, contexttype=t, window=5, context_scope=s)
    cont10 = Contexter(vectors, contexttype=t, window=10, context_scope=s)

    
    vector_vocabulary1 = cont1.process_data(sentences)
    vector_vocabulary5 = cont5.process_data(sentences)
    vector_vocabulary10 = cont10.process_data(sentences)
    #poor computah :()

    cont_dict = cont1.vocabt
#    print(cont_dict)
#    for v in cont_dict:
#        print(v, cont_dict[v])
    ###PPMI of co-occurence
    xpmi = cont1.PPMImatrix(cont_dict, documents)
#    for v in pmi:
#        print(v)
#        for n in pmi[v]:
#            print(n, pmi[v][n])
    
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

    #Initialize similarity class
    sim1 = Similarity(vector_vocabulary1, pmi = xpmi)
    sim5 = Similarity(vector_vocabulary5, pmi = xpmi)
    sim10 = Similarity(vector_vocabulary10, pmi = xpmi)

    with open('/home/usr1/git/dist_data/combined.csv') as f:
        with open('/home/usr1/git/dist_data/combined1.csv', 'w') as res:
            csv_wr = csv.writer(res, delimiter=',')
            for i, ln in enumerate(f):
                ln = ln.lower().rstrip().split(',')
                s1 = sim1.cosine_similarity(ln[0], ln[1])
                s5 = sim5.cosine_similarity(ln[0], ln[1])
                s10 = sim10.cosine_similarity(ln[0], ln[1])
                
                try:
                    o = float(s1)
                    oo = float(s5)
                    ooo = float(s10)

                except Exception as e:
                    continue

                csv_wr.writerow([ln[0], ln[1], ln[2], s1, s5, s10])

#PEARSON SPEARMAN TESTING
#################################################
    if teststuff:
        with open('/home/usr1/git/dist_data/combined1.csv') as f:
            humanv = []
            riv1 = []
            riv5 = []
            riv10 = []
            for i, ln in enumerate(f):
                ln = ln.rstrip().split(',')
#                print(ln)
                humanv.append(float(ln[2]))
                riv1.append(float(ln[3]))
                riv5.append(float(ln[4]))
                riv10.append(float(ln[5]))


        print(len(humanv), len(riv1), len(riv5), len(riv10))
        print(st.stats.spearmanr(humanv, riv1))
        print(st.pearsonr(humanv,riv1), '1\n')
        print(st.stats.spearmanr(humanv, riv5))
        print(st.pearsonr(humanv,riv5), '5\n')
        print(st.stats.spearmanr(humanv, riv10))
        print(st.pearsonr(humanv,riv10), '10\n')

#TSNE PLOTTING
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