'''
  Assumptions: 
	Shorter time horizons are often easier to predict with higher confidence.
    Frequency: Perhaps data is provided at a frequency that is too high to 
	 model or is unevenly spaced through time requiring resampling for 
	 use in some models.
    Outliers: Perhaps there are corrupt or extreme outlier values that need to
	 be identified and handled.
    Missing: Perhaps there are gaps or missing data that need to be 
	 interpolated or imputed.
'''

from mido import MidiFile

# multivariate LSTM forecasting
from numpy import array
from numpy import hstack
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

debug = True

def show_info(mid):
    print("Basic Info..")
    print("Number of tracks {}: ", len(mid.tracks))
    for i, track in enumerate(mid.tracks): 
        print('Track {}: {}'.format(i, track.name))
    print("Type: {}".format(mid.type))
    print("Length in seconds: {}".format(mid.length))

 	
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


def unitary_train(tracks):
    out_seq = np.zeros(len(tracks[0]))
    for i, track in enumerate(tracks):
        out_seq = array([track[len(track) - 1] for i in range(len(track))])
        # convert to [rows, columns] structure
        track = track.reshape((len(track), 1))
        out_seq = out_seq.reshape((len(out_seq), 1))


    # horizontally stack columns
    dataset = hstack(tracks)
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    print(X.shape, y.shape)
    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])


def main(): 
    # Read the file
    mid = MidiFile('midi_partitures/el_aguacate.mid')
    
    n_channels = 16
    seconds = mid.length
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = ticks_per_beat * 2 # (120 beats / 60 seconds). 
                                          # Default of 120 beats per minute.
    l = []
    for i, track in enumerate(mid.tracks):
        l.append(len(track))
    max_notes = max(l)

    tracks = np.full((n_channels, max_notes + 10000), 60)
    velocity = np.full((n_channels, max_notes + 10000), 64)
    time = np.full((n_channels, max_notes + 10000), 0)

    # Play the song ...
    current_time = 0.0
    i = 0
    for msg in mid.play():
        if (msg.type == 'note_on'):
            if msg.time != current_time:
                i = i + 1
                current_time = msg.time
            # type, channel, note, velocity, time.
            tracks[msg.channel][i] = msg.note
            velocity[msg.channel][i] = msg.velocity
            time[msg.channel][i] = msg.time
            # define input sequences.
    
    print(tracks)

if __name__ == "__main__": 
    mid = MidiFile('midi_partitures/el_aguacate.mid')
    
    n_channels = 16
    seconds = mid.length
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = ticks_per_beat * 2 # (120 beats / 60 seconds). 
                                          # Default of 120 beats per minute.
    l = []
    for i, track in enumerate(mid.tracks):
        print(len(track))