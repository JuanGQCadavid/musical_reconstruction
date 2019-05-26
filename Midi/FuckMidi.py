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

from mido import Message, MidiFile, MidiTrack

# multivariate LSTM forecasting
from numpy import array
from numpy import hstack
import numpy as np
#from tensorflow.keras import optimizers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import RepeatVector
#from tensorflow.keras.layers import TimeDistributed


debug = True

def show_info(mid):
    print("Basic Info..")
    print("Number of tracks {}: ", len(mid.tracks))
    for i, track in enumerate(mid.tracks): 
        print('Track {}: {}'.format(i, track.name))
    print("Type: {}".format(mid.type))
    print("Length in seconds: {}".format(mid.length))

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


def unitary_train(tracks):
    new_tracks = []
    for i, track in enumerate(tracks):
        track = np.array(track)
        #out_seq = array([track[len(track) - 1] for i in range(len(track))])
        # convert to [rows, columns] structure
        track = track.reshape((len(track), 1))
        new_tracks.append(track)

    # horizontally stack columns
    dataset = hstack(new_tracks)
    print(dataset.shape)
    
    # choose a number of time steps
    n_steps_in = 10
    n_steps_out = 1
    # convert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)

    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]

    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=300, verbose=1)
    return model


def readMidi(mid):
    n_channels = 16
    seconds = mid.length
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = ticks_per_beat * 2 # (120 beats / 60 seconds). 
                                          # Default of 120 beats per minute.
    l = []
    for i, track in enumerate(mid.tracks):
        l.append(len(track))
    max_notes = max(l)

    # define input sequences.

    tracks = np.full((n_channels, max_notes ), 60)
    velocity = np.full((n_channels, max_notes), 64)
    time = np.full((n_channels, max_notes), 0)

    # Play the song ...
    current_time = 0.0
    i = 0
    contador = 0
    model = None
    meta_msgs = []
    notes_msgs = []
    chanles_used = [False]*16

    for msg in mid(meta_messages=True):
        if msg.is_meta:
            meta_msgs.append(msg)
        if (msg.type == 'note_on'):
            
            if msg.time != current_time:
                i = i + 1
                current_time = msg.time
            # type, channel, note, velocity, time.
            if(not(chanles_used[msg.channel])):
                chanles_used[msg.channel] = True
            tracks[msg.channel][i] = msg.note
            velocity[msg.channel][i] = msg.velocity
            time[msg.channel][i] = msg.time
            contador = contador + 1

    for chanel in range(0,n_channels):
        if not(chanles_used[chanel]):
            tracks = np.delete(tracks, chanel)
            velocity = np.delete(velocity, chanel)
            time = np.delete(time, chanel)
    
    return tracks,velocity,time,meta_msgs


def readMidi_2(mid):

    tracks = mid.tracks

    #tracks = np.full((n_channels, max_notes ), 60)
    #velocity = np.full((n_channels, max_notes), 64)
    #time = np.full((n_channels, max_notes), 0)

    actual_time,acumulated_time = 0,0

    meta_data = []
    tracks = []
    no_note = 60

    for track in tracks:
        new_track = []
        new_meta = []
        actual_time,acumulated_time = 0,0
        for msg in track:
            if msg.is_meta:
                new_meta.append(msg)
            else:
                while True:
                    pass
                #if (actual_time + msg.time 





def main():
    # Read the file
    mid = MidiFile('midi_partitures/el_aguacate.mid')

    seconds = mid.length
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = ticks_per_beat

    print(seconds,ticks_per_beat,ticks_per_second)
    


    
    #for track in mid.tracks:
        #print(track)
        #print('*'*65)
        #for iduno in track:
            #print(iduno)

    #tracks,velocity,time,meta_msgs = readMidi(mid)

    #print(tracks.shape)
    #print(velocity.shape)
    #print(time.shape)


    

    '''
    # Write the song.
    file = MidiFile(type=1)
    for meta in enumerate(meta_msgs):
        print(meta)
    for i in range(0, n_channels):
        track_i = MidiTrack()
        for j in range(0, max_notes):
            track_i.append(Message('note_on', note=tracks[i][j], 
                                velocity=velocity[i][j], time=time[i][j]))
        
        file.tracks.append(track_i)
    
    file.print_tracks()
    file.save('fixed_' + mid.filename)
    print('wrote')

    '''
    '''# demonstrate prediction
    x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)'''

if __name__ == "__main__":
    main() 