#! /usr/bin/env python
# -*- coding: utf-8 -*-

import string 
import gensim

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from gensim.models.phrases import Phrases, Phraser
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file
from scipy.io import wavfile

def main():
    rate, data = wavfile.read('songs/hakuna_matata.wav')
    vocab = sorted(set(data))
    data = data.astype(str)
    data = [data[i:i + 25000] for i in range(0, len(data), 25000)]
    data = [x.tolist() for x in data]
    
    print('\nTraining word2vec...')
    word_model = gensim.models.Word2Vec(data, size=100, min_count=1, window=5, iter=100)
    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape
    print('Result embedding shape:', pretrained_weights.shape)

    def word2idx(word):
        return word_model.wv.vocab[word].index
    def idx2word(idx):
        return word_model.wv.index2word[idx]
    
    

def graph(rate, data):
    time = np.linspace(0, len(data)/rate, num=len(data))
    plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(time, data)
    plt.show()

def show_info(aname, a):
    print ("Array", aname)
    print ("shape:", a.shape)
    print ("dtype:", a.dtype)
    print ("min, max:", a.min(), a.max())
    print ()

if __name__ == "__main__": 
    main()
    