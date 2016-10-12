# Distributional Semantics
Distributional Semantics with Random Indexing

Optional LSA/HAL-like vectors (vocabulary*vocabulary co-occurrence matrix), no dimensionality reduction as of yet. Use at own risk.

CBOW or skip-gram to determine contexts. Default sliding window size is 1, can be changed.

Commands:
* "sim word1 word2" similarity between two words
* "top word" top 5 similar words
* "lsasim word1 word2" LSA/HAL-like similarity between two words
* "info" information about the data
* "info word" information about a word
* "save name" save current data as "name"
* "load name" to load saved data 
* "update path" update the current data with new data(.txt file)
* "help" to display all commands
