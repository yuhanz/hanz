import torch
import torch.nn as nn
import hanz
import sys
import math
import numpy as np
import pdb

file_name = 'examples/example-masked-self-attention.hanz'

modules, functions = hanz.parseHanz(file_name)
func = functions[0]
model = modules[0]

text = 'Hello world <eos> Hello torch <eos>'
dictionary = ['<eos>', 'hello', 'world', 'torch']
embedding_matrix = [[0.0, 0.0], [1.1, 1.1], [1.1, 2.1], [2.1, 1.1]]
tokens = list(map(lambda r: r.lower(), text.split(' ')))
embeddings = list(map(lambda t: embedding_matrix[dictionary.index(t)], tokens))
max_sentence_length = 20
dimension_of_embedding = 2
all_positional_encodings = list(map(lambda pos: list(map(lambda x: math.cos(- math.pi / 2 * (x % 2) + pos/10000**(int(x/2)*2/dimension_of_embedding)), range(0, dimension_of_embedding))), range(0, max_sentence_length)))


vector_to_word_model = nn.Sequential(nn.Linear(dimension_of_embedding, len(dictionary)), nn.Softmax(dim=0))

learning_rate = 0.0002
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(list(model.parameters()) + list(vector_to_word_model.parameters()), lr=learning_rate)

for (index, current_embedding) in enumerate(embeddings):
    previous_embeddings = embeddings[0:index+1]
    current_vectors = [current_embedding] * len(previous_embeddings)
    positional_encodings = all_positional_encodings[0:index+1]

    if index == 0:
        continue
    # pdb.set_trace()

    result = func(query_vector= torch.Tensor(current_vectors), positional_encoding= torch.Tensor(positional_encodings), word_embedding= torch.Tensor(previous_embeddings))
    masked_self_attention = torch.sum(result, 0)

    current_word_vector = torch.Tensor(current_embedding) + torch.Tensor(all_positional_encodings[index]) + masked_self_attention

    outcome = vector_to_word_model(current_word_vector)

    # get expected output
    current_token = tokens[index]
    word_index = dictionary.index(current_token)
    expected = torch.Tensor([0.0]* 4)
    expected[word_index] = 1.0
    # print('--result', result)
    print('--outcome', outcome)
    print('expected', expected)
    loss = torch.nn.L1Loss()(outcome, torch.Tensor(expected))
    print('loss', loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# pdb.set_trace()
