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
import random
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed


def model_vanilla(n_steps,n_features):
    global model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

def model_stacked_lstm(n_steps,n_features):
    global model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

def train_batch(notes):
    x,y = split_sequence(notes, n_steps)


    x = x.reshape((x.shape[0],x.shape[1],n_features))

    model.fit(x,y,epochs=epochs,verbose=verbose)

def split_sequence(sequence, n_steps):
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > (len(sequence) -1):
            break
        
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return array(X), array(y)

def ralph(notes,percent):
    broken_pos = [False] * len(notes)

    for i,note in enumerate(notes):
        ran  = random.random()

        if(ran < percent) and (i > 20):
            broken_pos[i] = True
            note.note = 0

        
    return notes,broken_pos

def write_midi(mid,my_tracks,diomio_number):
    file = MidiFile(type=mid.type)
    file.ticks_per_beat = mid.ticks_per_beat
    for track in my_tracks:
        track_i = MidiTrack()
        #for msg in track[2]:
            #track_i.append(msg)
        
        for meta_msg in track[0]:
            track_i.append(meta_msg)
        for note in track[1]:
            track_i.append(note)
        
        file.tracks.append(track_i)
        
    file.save('diomio_'+str(diomio_number)+'.mid')
    print('wrote')

def read_midi(mid):
    my_tracks = []
    for track in mid.tracks:
        meta_msg = []
        notes_msg = []
        msgs = []
        for msg in track:
            msgs.append(msg)
            if msg.type == 'note_on':
                notes_msg.append(msg)
            else:
                meta_msg.append(msg)
        
        my_tracks.append([meta_msg,notes_msg,msgs])
    
    return my_tracks

def predict(sequece):
    x = sequece.reshape((1,n_steps,n_features))
    yhat = model.predict(x,verbose=verbose)
    return yhat

def reparador_felix_jr(tracks):
    meta_msgs, notes_msgs, flags = tracks
    
    notes = []
    for msg in notes_msgs:
        notes.append(msg.note)

    notes = np.array(notes)

    for i,flag in enumerate(flags):
        if flag:
            train_batch(notes[0:i])
            note_predicted = predict(notes[i-n_steps:i])
            notes[i] = note_predicted
    
    tracks[0] = meta_msgs
    tracks[1] = notes.tolist()
    tracks[2] = flags

    return tracks

model = None
n_steps = 10 # n notes used to predict n features
n_features = 1 # Only one track
epochs = 200 # n passes through the dataset 
verbose = 1  #Show logs 

def main():
    global model, n_steps, n_features
    n_steps,n_features = 10,1
    diomio_number = '9'

    # Read the file
    mid = MidiFile('midi_partitures/el_aguacate.mid')
    
    
    my_tracks = read_midi(mid)

    #Dañar
    for track in my_tracks:
        notes = track[1]
        track[1],track[2] = ralph(notes,0.4)  
    
    #escribir
    write_midi(mid,my_tracks,'broken_'+diomio_number)
    model = model_stacked_lstm(n_steps,n_features)
    #Reparar
    for pos,track in enumerate(my_tracks):
        my_tracks[pos] = reparador_felix_jr(track)

    write_midi(mid,my_tracks,'repair_'+ diomio_number)
    

    # Write the song.



    '''# demonstrate prediction
    x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)'''

if __name__ == "__main__":
    main() 