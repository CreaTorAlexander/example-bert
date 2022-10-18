## Benefits from BERT over other models

- BERT produces word representations that are dynamically informed by the
words around them.

### Example

- “The man was accused of robbing a bank.”
- “The man went fishing by the bank of the river.”

Word2Vec would produce the same word embedding for the word "bank" in both sentences,
while under BERT the word embedding for "bank" would be different
  for each sentence.

### What is a word embedding?

Word Embedding = Feature Vector represntation of a word.


### Question
- What do the models learn?


Calculation
Layer Wise?
- Dim Reduction to show words which are close together in a 2D space

Self Similarity?
- Explanation 
Computing contextualization scores
As shown by Ethayarajh (2019), statistical measures such as word self-similarity can be used to describe the degree of word contextualization. Self-similarity is defined as "the average cosine
similarity of a word with itself across all the contexts in which it appears, where representations of the word are drawn from the same layer of a given model." 
We compute the self similarity score for all words in the corpus.


For visualization -> Dim Reduction and then 
- How they reduce 768 dim to x and y?
-> By this words with similar embeddings appear next to each other in the 2D Space
-> Projection for each models layer BERT 12 layers

