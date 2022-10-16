# First Steps with Bidirectional Encoder Represenations from Transformers (BERT)

## Based on the blog post of Christ McCormick

-> https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

## BERT is a pre trained language model

As it is pre trained it expects the input data in a specific format

1. A special token, [SEP], to mark the end of a sentence, or the separation between two sentences
2. A special token, [CLS], at the beginning of our text. This token is used for classification tasks, but BERT expects it no matter what your application is.
3. Tokens that conform with the fixed vocabulary used in BERT
4. The Token IDs for the tokens, from BERT’s tokenizer
5. Mask IDs to indicate which elements in the sequence are tokens and which are padding elements
6. Segment IDs used to distinguish different sentences
7. Positional Embeddings used to show token position within the sequence

Luckily the transformer interfaces takes care of the requirements
by using the tokenizer.encode_plus function

## Example Of Inputs

2 Sentence Input:
- [CLS] The man went to the store. [SEP] He bought a gallon of milk.
1 Sentence Input:
- [CLS] The man went to the store. [SEP]

The tokens [CLS] and [SEP] are always required

## Tokenization
text = "Here is the sentence I want embeddings for."

produces
`['[CLS]', 'here', 'is', 'the', 'sentence', 'i', 'want', 'em', '##bed', '##ding', '##s', 'for', '.', '[SEP]']`

"embedding" is represented as
`['em', '##bed', '##ding', '##s']`

After breaking the text into tokens, we have to convert the sentence from
a list of strings to a list of vocabulary indeces.

From here on, we'll use the below example sentence, which contains
two instances of the word "bank" with different meanings.

Example
- After stealing money from the bank vault, the bank robber was seen 
- fishing on the Mississippi river bank.
