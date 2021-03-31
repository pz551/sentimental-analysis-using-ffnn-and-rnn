import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from pathlib import Path 
import time
from tqdm import tqdm
from data_loader import fetch_data
from gensim.models import Word2Vec

unk = '<UNK>'


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers): # Add relevant parameters
        super(RNN, self).__init__()
        # Fill in relevant parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, 5)
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)  

    def forward(self, inputs): 
        #begin code
        hidden = torch.zeros(self.n_layers, self.hidden_dim)
        out, hidden = self.rnn(inputs, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        predicted_vector = self.softmax(out) # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
        #end code
        return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)

# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data):
    sentences = []
    vectorized_data = []
    for document, _ in data:
        sentences.append(document)
    model = Word2Vec(sentences, min_count=1)
    for document, y in data:
        vector = []
        for word in document:
            vector.append(model[word])
        vectorized_data.append((vector, y))
    return vectorized_data
    

def main(hidden_dim, n_layers, number_of_epochs): # Add relevant parameters
    print("Fetching data")
    train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    print("Fetched and indexed data")
    train_data = convert_to_vector_representation(train_data)
    valid_data = convert_to_vector_representation(valid_data)
    print("Vectorized data")
    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
    # Option 3 will be the most time consuming, so we do not recommend starting with this
   
    model = RNN(len(vocab), hidden_dim, n_layers) # Fill in parameters
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8) 
    print("Training for {} epochs".format(number_of_epochs))
    for epoch in range(number_of_epochs): # How will you decide to stop training and why
        # You will need further code to operationalize training, ffnn.py may be helpful
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model.forward(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data) # Good practice to shuffle order of validation data
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            #optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model.forward(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            #loss.backward()
            #optimizer.step()
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        # You may find it beneficial to keep track of training accuracy or training loss; 

        # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

        # You will need to validate your model. All results for Part 3 should be reported on the validation set. 
        # Consider ffnn.py; making changes to validation if you find them necessary