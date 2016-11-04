# Distributional Semantics
Distributional Semantics with Random Indexing


#Modules
WordSpaceModeller.py
* collections.defaultdict
* re
* random
* math
* sklearn.metrics.pairwise
* stemming.porter2.stem
* numpy

tester.py
* tsne (https://lvdmaaten.github.io/tsne/), modified by me for python 3.5.2
* matplotlib.pyplot
* csv
* scipystats

#Usage
Sample usage/testing can be found is tester.py

main.py contains a terminal interface for WordSpaceModeller.py

#Classes

DataReader:

    Reads data from .txt files and organizes them into sentence, creates a vocabulary and summarises word counts in each document.

    PARAMS:
        INIT:
            docsentences: organize sentences in [doc1[sents]doc2[sents]] (default: False)
            nums: convert numbers to NUM (default: True)
            percs: convert percentages to PERC (default: True)

    INPUT:
        preprocess_data: List of .txt files
        sentencizer: List of strings
        word_formatter: string

    OUTPUT:
        preprocess_data: List of sentences, list of words in vocabulary, dictionary of documents with wordcount in them
            sentencizer: list of sentences
            word_formatter: string


RandomVectorizer:

	Creates word and random vectors from a vocabulary(list of words)

    PARAMS:
		INIT:
        	dimensions: dimensionality of the random/word vector (default: 1024)
        	random_elements: how many random indices to insert +1's and -1's into in the random vector (default: 6)

    INPUT:
        vocabulary_vectorizer: List of words
        	random_vector: None

    OUTPUT:
        Dictionary of words in vocabulary with a word_vector and a random_vector


TermRelevance:

	Weights vector based on tf-idf
	https://en.wikipedia.org/wiki/Tf%E2%80%93idf

	Weighter.weight_setup for weight_setup

    SCHEMES:

    scheme 0: standard
        freq/doc_freq * log(N/n)

    scheme 1: log normalization
        1 + log10(freq/doc_freq) * log(N/n)

    scheme 2: double normalization
        0.5 + (0.5 * (freq/doc_freq))/(max_freq*(max_freq/doc_freq)))) * log(N/n)

    PARAMS:
        scheme: select a weighting scheme to use (default: 0)
			standard = 0
			log normalization = 1
			souble norm/augmented = 2

        smooth_idf: smooth the idf weight log(1+(N/n))(default: False)
        doidf: compute idf (default: True)

    INPUT:
        INIT:
            document_dict: dictionary of documents and their word count
            >>> dict{doc{word: count}}

        METHODS:
            weight: word/string and random vector, list of strings, dict{word: {random_vector: vector}}
            (weight calls the appropriate method below)
			weight_list: list of strings
			weight_vector: string, vector
			weight_dict: dict{word: {random_vector: vector}}
                
			tf: string
            idf: string

    OUTPUT:
        weight: Depends on data, see below
        weight_list: dict of weights
		weight_vector: vector
		weight_dict: dict


Contexter:

    Reads sentences/text and determines a words context, then performs vector addition.
    Takes pre-weighted vectors

    Contexter.data_info to get the data_info

    PARAMS:
		INIT:
		    contexttype: Which type of context to use (default: CBOW)
				CBOW = 0
				skipgram = 1
		    window: size of context, CBOW: how many words to take, skipgram: how many words to skip (default: 1)
		    contextscope: set context boundries (default: 2)
				sentence = 0
				document = 1
				corpus/entire text = 2
		    distance_weights: give weights to words based on distance (default: False) TODO TODO TODO
		    weights: do weighting in this class (default: False)
		        >>> dict{word: weight}
			readwrite: do vector additions as the data is read (default: True)
			savecontext: save the context words (default: False)
		preprocess_data:
			return_vectors: return vectors or a dictionary of context words (default: True)
			dopmi: compute PMI values (default: False, input = doc: {word: count})
		

    INPUT:
        INIT:
            vocabulary of word vectors or nothing( output is then {word: [words in context]}, if save_context = True )
            >>> dict{word: {word_vector: [word_vector], random_vector: [random_vector]}}

        METHODS:
            process_data: sentences/text
            read_contexts: sentence/text or list of sentences
            vector_addition: string1, string2
			PMI: dictionary of word: [words in context], dictionary of word count in documents(same input as in TermRelevance)

    OUTPUT:
        process_data: dictionary of {word: updated word_vectors}
        read_contexts: dictionary of {word: [words in context]}
        vectors_addition: word_vector


Similarity:

	Cosine similarities between vectors

    INPUT:
        INIT:
            vocabulary of words and their word_vectors
            >>> dict{word:[word_vector]}
			pmi: apply PMI weights (default: False)

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


		Australia Australian applied linguistics took as its target the applied linguistics of mother tongue teaching and teaching English to immigrants. 
		The Australia tradition shows a strong influence of continental Europe and of the USA, rather than of Britain. 
		Applied Linguistics Association of Australia (ALAA) was established at a national congress of applied linguists held in August 1976. 
		ALAA holds a joint annual conference in collaboration with the Association for Applied Linguistics in New Zealand (ALANZ). and applied components.


Datareader

	sentences
	['australia', 'australian', 'appli', 'linguist', 'took', 'as', 'it', 'target', 'the', 'appli', 'linguist', 'of', 'mother', 'tongu', 'teach', 'and', 'teach', 'english', 'to', 'immigr'] 
	['the', 'australia', 'NUM', 'tradit', 'show', 'a', 'strong', 'influenc', 'of', 'continent', 'europ', 'and', 'of', 'the', 'u.s.a.,', 'rather', 'NUM', 'than', 'of', 'britain'] 
	['appli', 'linguist', 'associat', 'of', 'australia', 'alaa', 'was', 'establish', 'at', 'a', 'nation', 'congress', 'of', 'appli', 'linguist', 'held', 'in', 'august', 'NUM'] 
	['alaa', 'hold', 'a', 'joint', 'annual', 'confer', 'in', 'collabor', 'with', 'the', 'associat', 'for', 'appli', 'linguist', 'in', 'new', 'zealand', 'alanz'] 

	vocabulary:
		{'joint', 'associat', 'it', 'alaa', 'strong', 'immigr', 'australia', 'than', 'establish', 'confer', 'tongu', 'at', 'collabor', 'hold', 'a', 'new', 'NUM', 'for', 'teach', 'target', 'august', 'show', 'linguist', 'took', 'australian', 'alanz', 'held', 'with', 'u.s.a.,', 'as', 'english', 'nation', 'annual', 'mother', 'britain', 'tradit', 'in', 'congress', 'europ', 'the', 'appli', 'of', 'and', 'rather', 'influenc', 'was', 'zealand', 'to', 'continent'}

	documents:
		defaultdict(<class 'dict'>, {'test_doc_5.txt': 
			defaultdict(<class 'int'>, {'of': 6, 'joint': 1, 'associat': 2, 'it': 1, 'alaa': 2, 'strong': 1, 'immigr': 1, 'than': 1, 'confer': 1, 'tongu': 1, 'at': 1, 'collabor': 1, 'hold': 1, 'a': 3, 'new': 1, 'NUM': 3, 'for': 1, 'teach': 2, 'target': 1, 'august': 1, 'show': 1, 'linguist': 5, 'took': 1, 'australian': 1, 'alanz': 1, 'held': 1, 'with': 1, 'u.s.a.,': 1, 'as': 1, 'english': 1, 'nation': 1, 'annual': 1, 'mother': 1, 'britain': 1, 'tradit': 1, 'in': 3, 'congress': 1, 'europ': 1, 'the': 4, 'appli': 5, 'rather': 1, 'influenc': 1, 'was': 1, 'zealand': 1, 'to': 1, 'continent': 1, 'establish': 1, 'and': 2, 'australia': 3})})

RandomVectorizer

	vectors:
		defaultdict(<class 'dict'>, {'rather': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'random_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])}, 'joint': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'random_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])} ...


Weighter

	weight_dict:
		defaultdict(<class 'dict'>, {'rather': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'random_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])}, 'joint': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'random_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])}

	weight_list:
		defaultdict(<class 'int'>, 
			{'continent': 0.010869565217391304, 'zealand': 0.010869565217391304, 'tradit': 0.010869565217391304, 'alaa': 0.021739130434782608, 
			'for': 0.010869565217391304, 'alanz': 0.010869565217391304, 'annual': 0.010869565217391304, 'alan': 0.010869565217391304, 
			'confer': 0.010869565217391304, 'intellig': 0.010869565217391304, 'it': 0.010869565217391304, 'comput': 0.010869565217391304, 
			'linguist': 0.05434782608695652, ...

Contexter

	vectors:
		{'rather': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'joint': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])

	PMImatrix:
		defaultdict(<class 'dict'>, {'of': {'congress': 1.1083394747888382, 'the': 0.5062794834608758, 'appli': 0.40936947045281946, 'continent': 1.1083394747888382,

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

#testing/benchmarks/results:
WordSim353 (http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)

Top values:
CBOW, log-tf, idf
dims = 3072, random ind = 6

rho = 3.32, 
r = 0.29



