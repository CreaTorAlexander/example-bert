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

from scipy.spatial.distance import cosine


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

text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."

marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

# exploration of BERTs vocabulary:r
# list(tokenizer.vocab.keys())[5000:5020]

# Map token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6}'.format(tup[0], tup[1]))


# Mark each of the 22 tokens as belonging to sentence "1"
# For one sentence input
# every word of sentence 1 assign 0's
# every word of sentence 2 assign 1's

segments_ids = [1] * len(tokenized_text)


# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# model is a deep neurol network with 12 layers!
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


# Lets evaluate BERT on our example text, and fetch hidden states of the 
# network

# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers. 
with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]

# The full set of hidden states for this model, stored in the object hidden_states

print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

# We wanna get rid of the batches dimension
# And grouped by tokens not layers

token_embeddings = torch.stack(hidden_states, dim=0)

token_embeddings.size()

# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)

token_embeddings.size()

# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)

token_embeddings.size()

# Now what to do with the hidden states?
# We want individual vectors for each of our tokens
# or single vector for whole sentence
# but for each token of our input we have 13 seperate vectors each of length 768
# we have to combine some of the layer vectors

# Stores the token vectors, with shape [22 x 768]
token_vecs_sum = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:

    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
    sum_vec = torch.sum(token[-4:], dim=0)
    
    # Use `sum_vec` to represent `token`.
    token_vecs_sum.append(sum_vec)

print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

for i, token_str in enumerate(tokenized_text):
  print (i, token_str)

print('First 5 vector values for each instance of "bank".')
print('')
print("bank vault   ", str(token_vecs_sum[6][:5]))
print("bank robber  ", str(token_vecs_sum[10][:5]))
print("river bank   ", str(token_vecs_sum[19][:5]))

# Calculate the cosine similarity between the word bank 
# in "bank robber" vs "river bank" (different meanings).
diff_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[19])

# Calculate the cosine similarity between the word bank
# in "bank robber" vs "bank vault" (same meaning).
same_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[6])

print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
print('Vector similarity for *different* meanings:  %.2f' % diff_bank)
