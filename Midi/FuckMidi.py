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
import copy
import math
import time


def model_vanilla(n_steps,n_features):
    global model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(LSTM(25, activation='relu' ))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

def model_stacked_lstm(n_steps,n_features):
    global model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(25, activation='relu'))
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

        if(ran < percent) and (i > (n_steps * 2)):
            broken_pos[i] = True
            note.note = 0

        
    return notes,broken_pos

def write_midi(mid,my_tracks,path):
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
        
    file.save(path)
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
    global model
    meta_msgs, notes_msgs, flags = tracks
    model = model_stacked_lstm(n_steps,n_features)
    
    notes = []
    realn = []
    for msg in notes_msgs:
        notes.append(msg.note)

    notes = np.array(notes)
    max_i = 0
    for i in flags:
        if i:
            max_i = max_i + 1
    print (max_i)
    actual_i = 0
    for i, flag in enumerate(flags):
        if flag:
            start = i - (n_steps*2) if i > n_steps else 0
            train_batch(notes[start:i])
            note_predicted = predict(notes[i - n_steps:i])[0][0]
            note_predicted = 0 if note_predicted < 0 else note_predicted
            notes[i] = round(note_predicted) if note_predicted < 127 else 127
            print(actual_i, '/' , max_i)
            actual_i = actual_i + 1

            notes_msgs[i].note = notes[i]

    return tracks

model = None
n_steps = 15 # n notes used to predict n features
n_features = 1 # Only one track
epochs = 100 # n passes through the dataset 
verbose = 0  #Show logs
damage_rate = 0.2 # Damage

def compare_midi(real,my_tracks,times):
    print('inside')

    for pos,track in enumerate(my_tracks):
        print("TRACK ", pos)
        p_meta_msgs, p_notes_msgs, p_flags = my_tracks[pos]
        r_meta_msgs, r_notes_msgs, r_flags = real[pos]
        acomulated_error = 0
        if len(p_notes_msgs) == 0:
            continue

        corrupted_counter = 0 
        for j,p_msg_note in enumerate(p_notes_msgs):
            r_msg_note = r_notes_msgs[j]
            
            if p_flags[j]:
                corrupted_counter = corrupted_counter + 1
                error_relativo = (math.fabs(r_msg_note.note-p_msg_note.note)/r_msg_note.note)*100
                print('Real ->', r_msg_note.note, ' Predicted ->',p_msg_note.note,'Error ->',error_relativo)
                acomulated_error += error_relativo
        
        print(acomulated_error,corrupted_counter)
        acomulated_error =  acomulated_error/corrupted_counter
        print('Error Relativo acomulado: ',acomulated_error)

        print('Time Training')
        time_acumulate = 0
        for i in range(1,len(times)):
            print('Time in track',i,' -> ',times[i])
            time_acumulate += times[i]
        
        print('Promedium ->',time_acumulate/(len(times)-1))



def main():
    global model, n_steps, n_features
    n_steps,n_features = 10,1
    diomio_number = '9'

    # Read the file
    mid = MidiFile('midi_partitures/happy.mid')

    my_tracks = read_midi(mid)
    real = copy.deepcopy(my_tracks)

    #Dañar
    for track in my_tracks:
        notes = track[1]
        track[1],track[2] = ralph(notes, damage_rate)
    
    #escribir
    path = mid.filename.split('.')
    path = path[0] + '_broken.' + path[1]
    write_midi(mid,my_tracks, path)

    model = model_stacked_lstm(n_steps,n_features)
    #Reparar

    times = []
    for pos,track in enumerate(my_tracks):
        print("TRACK ", pos)
        actual_time = time.time() 
        my_tracks[pos] = reparador_felix_jr(track)
        delta_time = time.time()  - actual_time
        print('Delta time ->', delta_time)
        times.append(delta_time)


        


    # Write the song.
    path = mid.filename.split('.')
    path = path[0] + '_fixed.' + path[1]
    write_midi(mid,my_tracks, path)

    compare_midi(real,my_tracks,times)




if __name__ == "__main__":
    main() 