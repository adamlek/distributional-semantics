# Distributional Semantics
Distributional Semantics with Random Indexing


#Modules
* collections.defaultdict
* re
* random
* math
* sklearn.metrics.pairwise
* stemming.porter2.stem
* numpy


#Classes

DataReader:

    Reads data from .txt files and organizes them into sentence, creates a vocabulary and summarises word counts in each document.

    INPUT:
        preprocess_data: List of .txt files
            sentencizer: Line of text
            propernamer: list of strings
            word_formatter: string

    OUTPUT:
        preprocess_data: List of sentences, list of words in vocabulary, dictionary of documents with wordcount in them
            sentencizer: list of sentences
            propernamer: sentence
            word_formatter: word


RandomVectorizer:

	Creates word and random vectors from a vocabulary(list of words)

    PARAMS:
        dimensions: dimensionality of the random/word vector (default: 1024)
        random_elements: how many random indices to insert +1's and -1's into in the random vector (default: 6)

    INPUT:
        vocabulary_vectorizer: List of words
        	random_vector: None

    OUTPUT:
        Dictionary of words in vocabulary with a word_vector and a random_vector


Weighter:

	Weights vector based on tf-idf
	https://en.wikipedia.org/wiki/Tf%E2%80%93idf

	Weighter.weight_setup for weight_setup

    SCHEMES:

    scheme 0: standard
        freq/doc_freq * log(N/n)

    scheme 1: log normalization
        1 + log10(freq/doc_freq) * log(N/n)

    scheme 2: double normalization
        0.5 + (0.5 * ((freq/doc_freq)/(max_freq*(max_freq/doc_freq)))) * log(N/n)

    PARAMS:
        scheme: select a weighting scheme to use (default: 0)
        smooth_idf: smooth the idf weight log(1+(N/n))(default: False)
        doidf: compute idf (default: True)

    INPUT:
        INIT:
            document_dict: dictionary of documents and their word count
            >>> dict{doc{word: count}}

        METHODS:
            weight: word/string, [random vector]
            weight_list: list of strings
                tf: string
                idf: string

    OUTPUT:
        weight: tf-idf weighted vector
        weight_list: dict of weights


Contexter:

    Reads sentences/text and determines a words context, then performs vector addition.

    Contexter.data_info to get the data_info

    PARAMS:
        contexttype: Which type of context, CBOW or skipgram (default: CBOW)
        window: size of context, CBOW: how many words to take, skipgram: how many words to skip (default: 1)
        sentences: set context boundry at sentence boundaries (default: True)
        distance_weights: give weights to words based on distance (default: False) TODO TODO TODO
		weights: do weighting in this class
			>>> dict{word: weight}

    INPUT:
        INIT:
            vocabulary of word vectors
            >>> dict{word: {word_vector: [word_vector], random_vector: [random_vector]}}

        METHODS:
            process_data: sentences/text
                read_contexts: sentence/text
                vector_addition: string1, string2

    OUTPUT:
        Dictionary of words in vocabulary with updated word_vector


Similarity:

	Cosine similarities between vectors

    INPUT:
        INIT:
            vocabulary of words and their word_vectors
            >>> dict{word:[word_vector]}

        METHODS:
            cosine_similarity:
                input: string1, string2
                output: cosine similarity between str1 and str2

            top:
                input: string
                output: top 5 most cosine similar words


DataOptions:

	Handling data, commands are save, load and info

    save
        input: filename, vocabulary{word: vector}, documents{doc: word_counts}, data_info{name, context, window, weights}, weight_setup
        output: filename.npz

    load:
        input: filename
        output:
            vocabulary: defaultdict(dict),
            documents: defaultdict(dict),
            data_info: defaultdict(dict)

    info
        input: optional(string, -weight string, -docs)
        output: documents, data_info, (word)



#Examples (from tester.py)

Text

		In 1950, Alan Turing published an article titled "Computing-Machinery and Intelligence". Computational linguistics has theoretical and applied components.


Datareader

	sentences
		[['in', 'NUM', 'alan', 'ture', 'publish', 'an', 'articl', 'titl', 'comput', 'machineri', 'and', 'intellig'], ['comput', 'linguist', 'has', 'theoret', 'and', 'appli', 'compon']]

	vocabulary:
		['in', 'NUM', 'alan', 'ture', 'publish', 'an', 'articl', 'titl', 'comput', 'machineri', 'and', 'intellig', 'linguist', 'has', 'theoret', 'appli', 'compon']

	documents:
		defaultdict(<class 'dict'>, {'test_doc_5.txt': defaultdict(<class 'int'>, {'publish': 1, 'intellig': 1, 'alan': 1, 'comput': 2, 'theoret': 1, 'has': 1, 'ture': 1, 'articl': 1, 'linguist': 1, 'machineri': 1, 'titl': 1, 'appli': 1, 'compon': 1, 'in': 1, 'NUM': 1, 'an': 1, 'and': 2})})


RandomVectorizer

	vectors:
		defaultdict(<class 'dict'>, {'publish': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'random_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])}, 'intellig': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) ... })


Weighter

	weight:
		[ 0.  0.  0. ...,  0.  0.  0.]

	weight_list:
		defaultdict(<class 'int'>, {'publish': 0.05263157894736842, 'intellig': 0.05263157894736842, 'alan': 0.05263157894736842, 'comput': 0.10526315789473684, 'theoret': 0.05263157894736842, 'has': 0.05263157894736842, 'ture': 0.05263157894736842, 'articl': 0.05263157894736842, 'linguist': 0.05263157894736842, 'machineri': 0.05263157894736842, 'titl': 0.05263157894736842, 'appli': 0.05263157894736842, 'compon': 0.05263157894736842, 'in': 0.05263157894736842, 'NUM': 0.05263157894736842, 'an': 0.05263157894736842, 'and': 0.10526315789473684})


Contexter

	word_vectors:
		{'publish': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'intellig': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) ... }

ROUGH sketches of sentencizer / propernamer

Sentencizer: 
        
	1. IF first symbol = lower, start append first symbol
		FOR SYMBOL IN LINE:
		2. if i+2 >= len(line):
		start exists:
			start has one entry => append start:end
			start has +1 entries => check for proper names => append start:end
	
	3. if symbol is uppercase:
		if symbols not followed by . (mrs., mr., sir. etc)
		start append symbol
	
	4. if symbol is ., ? or !:
		if i+2 is upper:
		if i-2/3 is lower:
		if start exists:
			start has one try => append start:'.!?'
			start has +1 entries => check for proper names => append start:'.!?'
	
	1.
	In 1950, Alan Turing published an article titled "Computing-Machinery and Intelligence".
	Sixteen years had Miss Taylor been in Mrs. Woodhouse's family and Emma likes it.
	In New York City the lions live. 
	=>
	['In', 'NUM', 'PN', 'publish', 'an', 'articl', 'titl', 'Comput', 'Machineri', 'and', 'Intellig']
	['Sixteen', 'year', 'had', 'PN', 'been', 'in', 'PN', 'famili', 'and', 'Emma', 'like', 'it']
	['In', 'PN', 'the', 'lion', 'live']
	
	2.
	The wedding was very much like other weddings, where the parties
	have no taste for finery or parade; and Mrs. Elton, from the
	particulars detailed by her husband, thought it all extremely shabby,
	=>
	['The', 'wed', 'was', 'veri', 'much', 'like', 'other', 'wed', 'where', 'the', 'parti']
	['have', 'no', 'tast', 'for', 'fineri', 'or', 'parad', 'and', 'PN', 'from', 'the']
	['particular', 'detail', 'by', 'her', 'husband', 'thought', 'it', 'all', 'extrem', 'shabbi']


propernamer:
	
	input: [w1, w2, ... wn]

	for w in input
	skip w1
		if w[0] is upper
		w is not last word
			if w+1[0] is upper
			del w
			del w+1
			w+2 is not last word
				if w+2[0] is upper
				del w+2

		insert PN at index of deleted w's
	
	



#main.py commands:
* "sim word1 word2" similarity between two words
* "top word" top 5 similar words
* "info" information about the data
* "info word" information about a word
* "info -docs" info about the documents
* "info -weight word" info about the weight of a word
* "save name" save current data as "name"
* "load name" to load saved data
* "set setting value" change value of context or window size
* "help" to display all commands

