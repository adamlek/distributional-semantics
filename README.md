# Distributional Semantics
Distributional Semantics with Random Indexing

Modules:
* numpy
* collections.defaultdict
* re
* random
* math
* sklearn.metrics.pairwise 
* stemming.porter2.stem

Classes:
RandomIndexing:
* process_data: Reads a .txt file and creates a vocubulary and a list of sentences
\t* sentence_formatter: Formats sentences and builds the vocabulary
\t* random_vector: Creates random vectors

Weighter:
*weigh: Weights a vector with tf-idf

Contexts:
* read_data: Reads a list of sentences and update vectors from CBOW or skip-gram contexts
\t* read_contexts: Read a sentence and determines word in context
\t* vector_addition: Adds vectors from context

Similarity:
* cosine_similarity: Calculates cosine similarity between two words
\t* dot: Dot product of two arrays
\t* cosine_measure: Calculates cosine similarity between two arrays
* top: Top 5 most cosine similar words

DataOptions:
* save: Save data
* load: Load data
* info: information about the data/words 

Extender:
* Nothing yet


CBOW or skip-gram to determine contexts. Default sliding window size is 1, can be changed.

Commands:
* "sim word1 word2" similarity between two words
* "top word" top 5 similar words

* "info" information about the data
* "info word" information about a word

* "save name" save current data as "name"
* "load name" to load saved data 

* "help" to display all commands