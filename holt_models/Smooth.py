#!/usr/bin/env python3.6
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
from numpy import array
from numpy import hstack
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib as mpl
from Utilities import Utilities
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

def song_plotter(rate,data):
    mpl.style.use('seaborn')
    time = np.linspace(0, len(data)/rate, num=len(data))
    plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(time, data)
    plt.show()

def compare_plotter(title,rate,test_data,preddicted_data):
    mpl.style.use('seaborn')
    time = np.linspace(0,len(test_data)/rate, num=len(test_data))

    plt.figure(2)
    plt.title(title)
    plt.plot(time,test_data, label='test data')
    plt.plot(time,preddicted_data, 'g', label='test data')
    plt.show()

def split_song_by_seconds(rate,data,seconds,start_time=0):
    split_point = int(start_time + (rate*seconds))

    first_part = data[start_time:split_point]
    second_part = data[split_point:]

    return first_part, split_point

def getSongCorrupted():
    global rate,original_data,tools

    rate, original_data = wavfile.read(song_path) #Read the song

    if len(original_data.shape) == 2:
        original_data = original_data[0:,0]

    data = (original_data.copy()).tolist()
    tools = Utilities(percent =percent, intervals = intervals,level=level)
    corrupted_with_none,data_broken = tools.corrupt(data)
    return corrupted_with_none,data_broken

def write_song(corrupted_with_none,title):
    corrupted_with_out_none = tools.reconstruction(corrupted_with_none)
    #print(corrupted_with_none)
    #song_plotter(rate,corrupted_with_out_none)
    wavfile.write(title, rate, np.array(corrupted_with_out_none,dtype=original_data.dtype))


def model_stacked_lstm(n_steps,n_features):
    global model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(25, activation='relu'))
    model.add(LSTM(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_batch(notes):
    x,y = split_sequence(notes, n_steps)
    x = x.reshape((x.shape[0],x.shape[1],n_features))

    #model.fit(x,y,epochs=epochs,verbose=verbose)
    for i in range(0,epochs):
        model.train_on_batch(x,y)
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def predict(sequece):
    x = sequece.reshape((1,n_steps,n_features))
    yhat = model.predict(x,verbose=verbose)
    
    return yhat


def fix_song(song_frecuencies,data_broken):
    song = np.array(song_frecuencies)
    before_i = 0


    max_i = 0
    for i in data_broken:
        if i:
            max_i = max_i + 1
    print (max_i)
    fixed_counter = 0
    for i,frec in enumerate(song):
        if data_broken[i] == True:
            print(fixed_counter,'/',max_i)
            start = 0 # i - (n_steps*2) if i > n_steps else 0
            train_batch(song[start:i])
            note = predict(song[i-n_steps:i])
            song[i] = note
            before_i = i
            fixed_counter += 1
            #Here its a bad note
    return song.tolist()

    

song_path = 'songs/hakuna_matata.wav'
rate = 0
original_data = None

percent = 0.1
intervals = 1
level = 1
tools = None

n_steps = 150
n_features = 1
verbose = 0
epochs = 100
model = None


def main():
    global model
    corrupted_with_none,data_broken = getSongCorrupted()

    title = song_path.split('.')
    title = title[0] + '_broken.' + title[1]
    
    middle_point = len(corrupted_with_none)
    treeQuarters = int(middle_point + middle_point/2)
    oneQuarter =  int(middle_point/16)

    print(middle_point,'->',treeQuarters)
    test = corrupted_with_none[0:oneQuarter]
    broken_test = data_broken[0:oneQuarter]
    
    write_song(test,title)

    model = model_stacked_lstm(n_steps,n_features)
    fixed_song = fix_song(test,broken_test)

    title = song_path.split('.')
    title = title[0] + '_fixed.' + title[1]
    write_song(fixed_song,title)

main()