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

#Usage
Sample usage can be found is tester.py

main.py contains a terminal interface for WordSpaceModeller.py

#Classes

DataReader:

    Reads data from .txt files and organizes them into sentence, creates a vocabulary and summarises word counts in each document.
    Categorizations:
        1, 1.2, 1/3, 1950 => NUM
        1%, 1.33% => PERC
        Jesus Christ, New York City => PN [Very aggressive/rough]

    PARAMS:
        preprocess_data:
            pns: convert propernames to PN (default: False)
            nums: convert numbers to NUM (default: True)
            percs: convert percentages to PERC (default: True)

    INPUT:
        preprocess_data: List of .txt files
            sentencizer: Line of text
            propernamer: Line of text
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
    Takes pre-weighted vectors

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
                read_contexts: sentence/text or list of sentences
                vector_addition: string1, string2

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

		line1: In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence".
		line2: Australia Australian applied linguistics took as its target the applied linguistics of mother tongue teaching and teaching English to immigrants. 
		The Australia tradition shows a strong influence of continental Europe and of the USA, rather than of Britain. 
		Applied Linguistics Association of Australia (ALAA) was established at a national congress of applied linguists held in August 1976. 
		ALAA holds a joint annual conference in collaboration with the Association for Applied Linguistics in New Zealand (ALANZ). and applied components.


Datareader

	sentences
		[['in', 'NUM', 'alan', 'ture', 'publish', 'an', 'articl', 'titl', 'comput', 'machineri', 'and', 'intellig'], 
		['australia', 'australian', 'appli', 'linguist', 'took', 'as', 'it', 'target', 'the', 'appli', 'linguist', 'of', 'mother', 'tongu', 'teach', 'and', 'teach', 'english', 'to', 'immigr'], 
		['the', 'australia', 'tradit', 'show', 'a', 'strong', 'influenc', 'of', 'continent', 'europ', 'and', 'of', 'the', 'usa', 'rather', 'than', 'of', 'britain'], 
		['appli', 'linguist', 'associat', 'of', 'australia', 'alaa', 'was', 'establish', 'at', 'a', 'nation', 'congress', 'of', 'appli', 'linguist', 'held', 'in', 'august', 'NUM'], 
		['alaa', 'hold', 'a', 'joint', 'annual', 'confer', 'in', 'collabor', 'with', 'the', 'associat', 'for', 'appli', 'linguist', 'in', 'new', 'zealand', 'alanz']]

	vocabulary:
		{'alan', 'ture', 'continent', 'tradit', 'alaa', 'for', 'alanz', 'annual', 'confer', 'intellig', 'it', '.', 'linguist', 'usa', 'publish', 'comput', 'in', 'tongu', 'establish', 'at', 'took', 
		'collabor', 'articl', 'as', 'new', 'strong', 'mother', 'was', 'titl', 'appli', 'NUM', 'associat', 'english', 'congress', 'britain', 'joint', 'to', 'show', 'nation', 'of', 'hold', 
		'influenc', 'august', 'teach', 'rather', 'immigr', 'held', 'zealand', 'australian', 'europ', 'with', 'and', 'target', 'an', 'a', 'the', 'australia', 'than', 'machineri'}
		(ture == turing, stemmer is pretty aggressive with proper names)

	documents:
		defaultdict(<class 'dict'>, {'test_doc_4.txt': 
			defaultdict(<class 'int'>, {'alan': 1, 'zealand': 1, 'tradit': 1, 'alaa': 2, 'for': 1, 'alanz': 1, 'annual': 1, 'ture': 1, 'confer': 1, 'intellig': 1, 'it': 1, 'linguist': 5, 'usa': 1, 
			'tongu': 1, 'establish': 1, 'in': 4, 'took': 1, 'collabor': 1, 'english': 1, 'articl': 1, 'as': 1, 'hold': 1, 'strong': 1, '.': 5, 'publish': 1, 'was': 1, 'titl': 1, 'appli': 5, 'congress': 1, 
			'NUM': 2, 'associat': 2, 'comput': 1, 'the': 4, 'britain': 1, 'joint': 1, 'new': 1, 'to': 1, 'show': 1, 'of': 6, 'influenc': 1, 'august': 1, 'teach': 2, 'rather': 1, 'held': 1, 
			'machineri': 1, 'continent': 1, 'australian': 1, 'europ': 1, 'with': 1, 'and': 3, 'target': 1, 'an': 1, 'a': 3, 'at': 1, 'mother': 1, 'australia': 3, 'than': 1, 'immigr': 1, 'nation': 1})})

RandomVectorizer

	vectors:
		defaultdict(<class 'dict'>, {'continent': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'random_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])}, 
			'australian': {'word_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'random_vector': array([ 0.,  0.,  0., ...,  0.,  0.,  0.])}, ...


Weighter

	weight_vector:
		[ 0.  0.  0. ...,  0.  0.  0.]

	weight_list:
		defaultdict(<class 'int'>, 
			{'continent': 0.010869565217391304, 'zealand': 0.010869565217391304, 'tradit': 0.010869565217391304, 'alaa': 0.021739130434782608, 
			'for': 0.010869565217391304, 'alanz': 0.010869565217391304, 'annual': 0.010869565217391304, 'alan': 0.010869565217391304, 
			'confer': 0.010869565217391304, 'intellig': 0.010869565217391304, 'it': 0.010869565217391304, 'comput': 0.010869565217391304, 
			'linguist': 0.05434782608695652, ...

Contexter

	vectors:
		{'continent': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'zealand': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), 'tradit': array([ 0.,  0.,  0., ...,  0.,  0.,  0.]),

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

