
import gensim
import torch
import torch.nn as nn

# https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
# Load word2vec pre-train model
model = gensim.models.Word2Vec.load('./model/test.model')
weights = torch.FloatTensor(model.wv.vectors)


# Build nn.Embedding() layer
embedding = nn.Embedding.from_pretrained(weights)
embedding.requires_grad = False


# Query
query = 'to'
query_id = torch.tensor(model.wv.key_to_index[query]) # Get the index of the query word

# get your query vector 
vec = torch.tensor(model.wv.get_vector(query, norm=True))
print(vec)


# get your query vector by id (query_id)
embedding_vector = embedding(query_id)
print(embedding_vector)