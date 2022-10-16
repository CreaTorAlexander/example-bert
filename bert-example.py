import torch
from transformers import BertTokenizer, BertModel

# transformers provides a number of classes for applying BERT to 
# different tasks (token classification, text classification)

# BertModel good choice to extract embeddings

# OPTIONAL: if you want to have more information on whats
# happening
import logging
# logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# % matplotlib inline

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# BERT is pretrained so it expects input in a specific format

# 1. special token [SEP], to mark the end of a sentence, or the 
# seperation between two senteces
# 2. special token [CLS], at the beginning of our text, used for
# classification tasks, but BERT expexts it no matter what your
# application is.
# 3. Tokens that conform with the fixed vocabulary used in BERT
# 4. The Token IDs for the tokens, from BERT's tokenizer
# 5. Mask IDs to indicate which elements in the sequence are tokens
# and which are padding elements
# 6. Segment IDs used to distinguish different sentences
# 7. Positional Embeddings used to show token position within the sequence


# As input can BERT take either one or two sentences, and uses the 
# special token [SEP] to differentiate them. [CLS] token always appears
# at the start of the text

# BOTH Tokens are always required

# Examples
# 2 Sentence Input:
# [CLS] The man went to the store. [SEP] He bought a gallon of milk.
# 1 Sentence Input:
# [CLS] The man went to the store. [SEP]

# Tokenization own way

text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

