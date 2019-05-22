# univariate stacked lstm example
from numpy import array
import numpy as np 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from scipy.io import wavfile

# Hiper parameters.
# 1 if mono. 2 if stereo. 
n_features = 1
# Choose a number of time steps. In our use case this is the WAV file rate.
n_steps = 3 # Overriden below.

max_sentence_len = 25000 

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

# define input sequence
rate, raw_seq = wavfile.read('songs/hakuna_matata.wav')
raw_seq = raw_seq[np.logical_not(np.isnan(raw_seq))]
raw_seq = raw_seq.astype(int)

# choose a number of time steps
n_steps = int(len(raw_seq) / rate)

# sample
raw_seq = raw_seq[10000:11250] # random sample. dev purposes.
print(raw_seq)

# split into samples
X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=8, verbose=1)
# demonstrate prediction
x_input = raw_seq[:n_steps]
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)