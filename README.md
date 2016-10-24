# Distributional Semantics
Distributional Semantics with Random Indexing

CBOW or skip-gram to determine contexts, default window size is 1.

main.py commands:
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

Modules:
* collections.defaultdict
* re
* random
* math

* sklearn.metrics.pairwise 
* stemming.porter2.stem
* numpy

Classes:

DATA CLASSES

DataReader:

Reads data from .txt files and organizes them into sentences.

        PARAMS:
            Numerize: NULL ATM (default: False)
        
        INPUT:
            preprocess_data: List of .txt files
            sentencizer: Line of text
            word_formatter: string
                
        OUTPUT:
            List of sentences, list of words in vocabulary, dictionary of documents with wordcount in them

RandomVectorizer:

Creates word and random vectors from a vocabulary(list of words)

    PARAMS:
        dimensions: dimensionality of the random/word vector (default: 2048)
        random_elements: how many random indices to insert +1's and -1's into in the random vector (default: 6)

    INPUT:
        vocabulary_vectorizer: List of words
        random_vector: None

    OUTPUT:
        Dictionary of words in vocabulary with a word_vector and a random_vector

Weighter:

Weights vector based on tf-idf

    PARAMS:
        tf_log: add log-normalization to term frequency (default: False)
        tf_doublenorm: add double normalization to term frequency (default: False)
        idf: compute idf (default: True)
        
    INPUT:
        INIT: scheme: weighting scheme, arbitrary atm
            document_dict: dictionary of documents and their word count

        METHODS:        
        weight: word, random vector

        TODO:
            weight_object: dictionary of words and random vectors

    OUTPUT:
        tf-idf weighted vector

Contexts:

Reads sentences/text and determines a words context, then performs vector addition

    PARAMS:
        contexttype: Which type of context, CBOW or skipgram (default: CBOW)
        window: size of context, CBOW: how many words to take, skipgram: how many words to skip (default: 1)
        sentences: set context boundry at sentence boundaries (default: True)
        distance_weights: give weights to words based on distance (default: False) TODO TODO TODO

    INPUT:
        INIT: 
            vocabulary of word vectors
        METHODS:
            process_data: sentences/text
            read_contexts: sentence/text
            vector_addittion: word vector, random vector

    OUTPUT:
        Dictionary of words in vocabulary with a updated word_vector
        Dictionary of data_info: name: x, context: y, window: n, weights: m

USER QUERY/MODEL OUTPUT CLASSES

Similarity:

Cosine similarities between vectors

    INPUT:
        INIT: vocabulary of words and their word_vectors

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
        input: filename
        output: filename.npz

    load:
        input: filename
        output:
            vocabulary: defaultdict(dict),
            documents: defaultdict(dict),
            data_info: defaultdict(dict)

    info
        input: optional(string, -weight string, -docs)
        output: info about data or word if supplied


