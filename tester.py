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
    save = False

#    dataset1 = ['/home/usr1/git/dist_data/test_doc_5.txt']
#    dataset1 = ['/home/usr1/git/dist_data/austen-emma.txt', '/home/usr1/git/dist_data/austen-persuasion.txt', '/home/usr1/git/dist_data/austen-sense.txt']
#    dataset1 = ['/home/usr1/git/dist_data/reut1.txt', '/home/usr1/git/dist_data/reut2.txt']
    dataset1 = ['/home/usr1/git/dist_data/formatted2.txt', '/home/usr1/git/dist_data/reut1.txt', '/home/usr1/git/dist_data/reut2.txt']
#    dataset1 = ['/home/usr1/git/dist_data/formatted2.txt']

#DATAREADER
##################################################
    dr = DataReader(docsentences=False)
    #read the file dataset1 and output all sentences, all words, and information about word count/documents
    sentences, vocabulary, documents = dr.preprocess_data(dataset1)


#    for sent in sentences:
#        print(sent, '\n')
    print('sentences:\t', len(sentences))
    t = 0
    for v in documents:
        t += sum(documents[v].values())
    print('total tokens:\t', t)
    print('total types:\t', len(vocabulary))

    print('reading file done\n')
##################################################

    #SETTINGS
    w, t, s = 1, 1, 2
    d, r = 2024, 16
    si1, si2, si3 = 3, 5, 10
    print('weighting:\t', w, t, s)
    print('vectors:  \t', d, r)
    print('sizes:    \t', si1, si2, si3)
##RANDOMVECTORIZER
###################################################
    rv = RandomVectorizer(dimensions=d, random_elements=r)
    #create word and random vectors for the strings in vocabulary
    vectors = rv.vocabulary_vectorizer(vocabulary)
###################################################
#
##WEIGHTER
###################################################
    #init Weighter, with scheme 0 and don't do idf
    tr = TermRelevance(documents, scheme=w, doidf=True, smooth_idf=False)

    #weight the dictionary of vectors
    vectors = tr.weight(vectors)
##################################################
#
##CONTEXTER
##################################################
    #Init Contexter
    cont1 = Contexter(vectors, contexttype=t, window=si1, context_scope=s)
    cont5 = Contexter(vectors, contexttype=t, window=si2, context_scope=s)
    cont10 = Contexter(vectors, contexttype=t, window=si3, context_scope=s)

    vector_vocabulary1 = cont1.process_data(sentences)
    vector_vocabulary5 = cont5.process_data(sentences)
    vector_vocabulary10 = cont10.process_data(sentences)
    #poor computah :()

#    cont_dict1 = cont1.vocabt
#    cont_dict5 = cont5.vocabt
#    cont_dict10 = cont10.vocabt

#    ###PPMI of co-occurence
#    xpmi1 = cont1.PPMImatrix(cont_dict1, documents)
#    xpmi5 = cont5.PPMImatrix(cont_dict5, documents)
#    xpmi10 = cont10.PPMImatrix(cont_dict10, documents)

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
#    sim1 = Similarity(vector_vocabulary1, pmi = xpmi1)
#    sim5 = Similarity(vector_vocabulary5, pmi = xpmi5)
#    sim10 = Similarity(vector_vocabulary10, pmi = xpmi10)
    sim1 = Similarity(vector_vocabulary1)
    sim5 = Similarity(vector_vocabulary5)
    sim10 = Similarity(vector_vocabulary10)

    if teststuff:
        humanv = []
        riv1 = []
        riv5 = []
        riv10 = []
        with open('/home/usr1/git/dist_data/combined.csv') as f:
                for i, ln in enumerate(f):
                    ln = ln.lower().rstrip().split(',')

                    try:
                        riv1.append(float(sim1.cosine_similarity(ln[0], ln[1])))
                        riv5.append(float(sim5.cosine_similarity(ln[0], ln[1])))
                        riv10.append(float(sim10.cosine_similarity(ln[0], ln[1])))
                        humanv.append(float(ln[2]))

                    except Exception as e:
                        continue

        print(len(humanv), len(riv1), len(riv5), len(riv10))
        print(st.stats.spearmanr(humanv, riv1))
        print('pearson r, p-val', st.pearsonr(humanv,riv1), si1, '\n')
        print(st.stats.spearmanr(humanv, riv5))
        print('pearson r, p-val', st.pearsonr(humanv,riv5), si2, '\n')
        print(st.stats.spearmanr(humanv, riv10))
        print('pearson r, p-val', st.pearsonr(humanv,riv10), si3, '\n')

#PEARSON SPEARMAN TESTING
#################################################
    if save:
        with open('/home/usr1/git/dist_data/combined1.csv') as f:
            csv_w = csv.writer(f, delimiter=',')
            for i, v in humanv:
                scv_w.writerow(v, riv1[i], riv5[i], riv10[i])


#TSNE PLOTTING
##################################################
    if plotting:
        ar = []
        lbs = []
        for i, v in enumerate(vector_vocabulary5):
            if i%100 == 0:
                ar.append(vector_vocabulary5[v])
                lbs.append(v)

        Y = tsne.tsne(np.array(ar), 2, 50, 20.0)
        fig, ax = plt.subplots()
        ax.scatter(Y[:,0], Y[:,1], 20)

        for i, name in enumerate(lbs):
            ax.annotate(name, (Y[i][0], Y[i][1]))

        plt.show()

if __name__ == '__main__':
    main()