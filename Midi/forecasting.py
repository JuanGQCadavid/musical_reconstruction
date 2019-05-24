'''
  Assumptions: 
	Shorter time horizons are often easier to predict with higher confidence.
    Frequency: Perhaps data is provided at a frequency that is too high to 
	 model or is unevenly spaced through time requiring RESAMPLING for 
	 use in some models.
    Outliers: Perhaps there are corrupt or extreme outlier values that need to
	 be identified and handled.
    Missing: Perhaps there are gaps or missing data that need to be 
	 interpolated or imputed.
'''

# univariate stacked lstm example
from numpy import array
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from scipy.io import wavfile

# Hiper parameters.
# 1 if mono. 2 if stereo. 
n_features = 1
# Choose a number of time steps. In our use case this is the WAV file rate.
n_steps = 3 # Overriden below.

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
n_steps = 1

# sample
#raw_seq = raw_seq # random sample. dev purposes.

# split into samples
X = raw_seq[0:1323000] #split_sequence(raw_seq, n_steps)
y = raw_seq[1323000:1345050]
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(X.shape)

# define model
model = Sequential()
model.add(LSTM(1, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(10, activation='relu'))
model.add(Dense(1))
adam_optimizer = optimizers.Adam()
model.compile(optimizer=adam_optimizer, loss='mse')

# fit model
model.fit(X, y, epochs=8, batch_size=1, verbose=1)
# demonstrate prediction
x_input = raw_seq[:n_steps]
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)