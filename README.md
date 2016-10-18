# Distributional Semantics
Distributional Semantics with Random Indexing

CBOW or skip-gram to determine contexts, default window size is 1.

Commands:
* "sim word1 word2" similarity between two words
* "top word" top 5 similar words

* "info" information about the data
* "info word" information about a word

* "save name" save current data as "name"
* "load name" to load saved data 
* "set setting value" change value of context or window size

* "help" to display all commands

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
	* sentence_formatter: Formats sentences and builds the vocabulary
	* random_vector: Creates random vectors

Weighter:
* weight: Weights a vector with tf-idf

Contexts:
* read_data: Reads a list of sentences and update vectors from CBOW or skip-gram contexts
	* read_contexts: Read a sentence and determines word in context
	* vector_addition: Adds vectors from context

Similarity:
* cosine_similarity: Calculates cosine similarity between two words
	* dot: Dot product of two arrays
	* cosine_measure: Calculates cosine similarity between two arrays
* top: Top 5 most cosine similar words

DataOptions:
* save: Save data
* load: Load data
* info: information about the data/words 

Extender:
* Nothing yet


