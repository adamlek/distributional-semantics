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


def main():

    ### CONTENT OF test_doc_4.txt
    #In 1950, Alan Turing published an article titled "Computing-Machinery and Intelligence". Sixteen years had Miss Taylor been in Mrs. Woodhouse's family and Emma likes it.
    #In New York City the lions live.
    #Australia Australian applied linguistics took as its target the applied linguistics of mother tongue teaching and teaching English to immigrants.
    #> The Australia tradition shows a strong influence of continental Europe and of the USA, rather than of Britain.
    #> Applied Linguistics Association of Australia (ALAA) was established at a national congress of applied linguists held in August 1976.
    #> ALAA holds a joint annual conference in collaboration with the Association for Applied Linguistics in New Zealand (ALANZ).
    dataset1 = ['/home/usr1/git/dist_data/test_doc_4.txt']
    
#DATAREADER
##################################################
    dr = DataReader()
    #read the file dataset1 and output all sentences, all words, and information about word count/documents
    sentences, vocabulary, documents = dr.preprocess_data(dataset1)

    for sent in sentences:
        print(sent, '\n')

    #trasform a string into sentences
    c, etc = dr.sentencizer('Australia Australian applied linguistics took as its target the applied linguistics of mother tongue teaching and teaching English to immigrants. The Australia tradition shows a strong influence of continental Europe and of the U.S.A., rather than of Britain. Applied Linguistics Association of Australia (ALAA) was established at a national congress of applied linguists held in August 1976. ALAA holds a joint annual conference in collaboration with the Association for Applied Linguistics in New Zealand (ALANZ).')
#    for sent in c:
#        print(sent, '\n')
##################################################    
    
#RANDOMVECTORIZER
##################################################
    rv = RandomVectorizer()
    #create word and random vectors for the strings in vocabulary
    vectors = rv.vocabulary_vectorizer(vocabulary)
##################################################    

#WEIGHTER
##################################################
    #init Weighter, with scheme 0 and don't do idf
    wgt = Weighter(documents, scheme=0, doidf=False)

    #list of weights for all words in vocabulary
    weight_list = wgt.weight_list(vocabulary)

    #update the vector indiviually
    for item in vectors:
        vectors[item]['random_vector'] = wgt.weight_vector(item, vectors[item]['random_vector'])
    
    #weight the dictionary vectors
    weight_dict = wgt.weight_dict(vectors)
#################################################

#CONTEXTER
#################################################
    #Init Contexter
    cont = Contexter(vectors, contexttype='CBOW', window=1, sentences=True, weights=weight_list)

    #update vectors (the variable from Contexter initialization)
    #output: updated vectors
    vector_vocabulary = cont.process_data(sentences)

    #read word contexts from a list of sentences
    #output: {word: [words in context]}
    contxts = cont.read_contexts(sentences)
#    for w in contxts:
#        print(w, contxts[w])
##################################################

#DATAOPTIONS
##################################################
    # initialize DataOptions for saving/extracting information
    do = DataOptions()
    # save the data, needs some more testing ^^
    # do.save('tester', vector_vocabulary, documents, cont.data_info, wgt.weight_setup)
##################################################
    
#SIMILARITY
##################################################
    #Initialize similarity class
    sim = Similarity(vector_vocabulary)
    #sim between 'the' and 'and'
    simsim = sim.cosine_similarity('the', 'and')
    #top 5 similar words to 'the'
    simtop = sim.top_similarity('the')
##################################################

if __name__ == '__main__':
    main()