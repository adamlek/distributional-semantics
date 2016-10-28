# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:51:47 2016
@author: Adam Ek
"""
from WordSpaceModeller import DataReader
from WordSpaceModeller import RandomVectorizer
from WordSpaceModeller import Weighter
from WordSpaceModeller import Contexter
from WordSpaceModeller import Similarity
from WordSpaceModeller import DataOptions
import tsne
import matplotlib.pyplot as plt
import numpy as np


def main():
    plotting = False

    ### CONTENT OF test_doc_4.txt
    #In 1950, Alan Turing published an article titled "Computing-Machinery and Intelligence". Sixteen years had Miss Taylor been in Mrs. Woodhouse's family and Emma likes it.
    #In New York City the lions live.
    #Australia Australian applied linguistics took as its target the applied linguistics of mother tongue teaching and teaching English to immigrants.
    #> The Australia tradition shows a strong influence of continental Europe and of the USA, rather than of Britain.
    #> Applied Linguistics Association of Australia (ALAA) was established at a national congress of applied linguists held in August 1976.
    #> ALAA holds a joint annual conference in collaboration with the Association for Applied Linguistics in New Zealand (ALANZ).
    dataset1 = ['/home/usr1/git/dist_data/austen-emma.txt', '/home/usr1/git/dist_data/austen-sense.txt']
#    dataset1 = ['/home/usr1/git/dist_data/test_doc_3.txt']

#DATAREADER
##################################################
    dr = DataReader()
    #read the file dataset1 and output all sentences, all words, and information about word count/documents
    sentences, vocabulary, documents = dr.preprocess_data(dataset1)

#    for sent in sentences:
#        print(sent, '\n')

    #trasform a string into sentences
    c, etc = dr.sentencizer('Australia Australian applied linguistics took as its target the applied linguistics of mother tongue teaching and teaching English to immigrants. The Australia tradition shows a strong influence of continental Europe and of the U.S.A., rather than of Britain. Applied Linguistics Association of Australia (ALAA) was established at a national congress of applied linguists held in August 1976. ALAA holds a joint annual conference in collaboration with the Association for Applied Linguistics in New Zealand (ALANZ).')
#    for sent in c:
#        print(sent, '\n')
    print('reading file done')
##################################################

#RANDOMVECTORIZER
##################################################
    rv = RandomVectorizer(dimensions=2024)
    #create word and random vectors for the strings in vocabulary
    vectors = rv.vocabulary_vectorizer(vocabulary)
    print('vectoring done')
##################################################

#WEIGHTER
##################################################
    #init Weighter, with scheme 0 and don't do idf
    wgt = Weighter(documents, scheme=1, doidf=True, smooth_idf=True)

    #list of weights for all words in vocabulary
#    weight_list = wgt.weight_list(vocabulary)

    #update the vector indiviually
#    for item in vectors:
#        vectors[item]['random_vector'] = wgt.weight_vector(item, vectors[item]['random_vector'])

    #weight the dictionary vectors
    w_vector_vocab = wgt.weight(vectors)
    print('weighting done')
#################################################

#CONTEXTER
#################################################
    #Init Contexter
    t = 'CBOW'
    cont1 = Contexter(w_vector_vocab, contexttype=t, window=1, sentences=False)
    cont3 = Contexter(w_vector_vocab, contexttype=t, window=3, sentences=False)
    cont5 = Contexter(w_vector_vocab, contexttype=t, window=5, sentences=False)
    cont10 = Contexter(w_vector_vocab, contexttype=t, window=10, sentences=False)

    #update vectors (the variable from Contexter initialization)
    #output: updated vectors
#    print([word for li in sentences for word in li])
    vector_vocabulary1 = cont1.process_data(sentences)
    vector_vocabulary3 = cont3.process_data(sentences)
    vector_vocabulary5 = cont5.process_data(sentences)
    vector_vocabulary10 = cont10.process_data(sentences)

    #read word contexts from a list of sentences
    #output: {word: [words in context]}
#    contxts = cont.read_contexts(sentences)
#    for w in contxts:
#        print(w, contxts[w])
#    print('reading contexts done', type(vector_vocabulary))
##################################################

#DATAOPTIONS
##################################################
    # initialize DataOptions for saving/extracting information
#    do = DataOptions()
    # save the data, needs some more testing ^^
    # do.save('tester', vector_vocabulary, documents, cont.data_info, wgt.weight_setup)
##################################################

#SIMILARITY
##################################################
    #Initialize similarity class
#    print(vector_vocabulary)
    sim1 = Similarity(vector_vocabulary1)
    sim3 = Similarity(vector_vocabulary3)
    sim5 = Similarity(vector_vocabulary5)
    sim10 = Similarity(vector_vocabulary10)
    #sim between 'the' and 'and'
    words1 = ['he', 'when', 'the', 'talk']
    words2 = ['she', 'where', 'and', 'speak']
    for w1, w2 in zip(words1, words2):
        print(w1, w2)
        print(sim1.cosine_similarity(w1, w2))
        print(sim3.cosine_similarity(w1, w2))
        print(sim5.cosine_similarity(w1, w2))
        print(sim10.cosine_similarity(w1, w2))
    #top 5 similar words to 'the'
#    simtop = sim.top_similarity('the')
##################################################

#PLOTTING
##################################################
    if plotting:
        arrs = np.array([vector_vocabulary10[arr] for arr in vector_vocabulary10])
        Y = tsne.tsne(arrs, 2, 50, 20.0)
        plt.scatter(Y[:,0], Y[:,1], 20)
        plt.show()

if __name__ == '__main__':
    main()