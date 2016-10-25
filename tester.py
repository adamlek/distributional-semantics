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
    dataset1 = ['/home/usr1/git/dist_data/test_doc_5.txt']

    dr = DataReader()
    sentences, vocabulary, documents = dr.preprocess_data(dataset1)

    rv = RandomVectorizer()
    vectors = rv.vocabulary_vectorizer(vocabulary)

    wgt = Weighter(documents, scheme=0, doidf=False)

    weight_list = wgt.weight_list(vocabulary)

    cont = Contexter(vectors, contexttype='CBOW', window=2, weights=weight_list)

    vector_vocabulary = cont.process_data(sentences)

if __name__ == '__main__':
    main()