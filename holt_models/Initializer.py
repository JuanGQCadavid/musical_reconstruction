from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib as mpl
from utilities import Utilities

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

def main():
    song_path = 'songs/hakuna_matata.wav'
    rate, data = wavfile.read(song_path) #Read the song

    print('rate -> ', rate, ' Data len -> ',len(data))

    song_plotter(rate,data)

    tools = Utilities(percent =0.1, intervals = rate/2, ratio = rate ,level=1)
    corrupted_with_none = tools.corrupt(data)

    song_plotter(rate,corrupted_with_none)

    corrupted_with_out_none = tools.reconstruction(corrupted_with_none)

    wavfile.write('songs/odebrecht_hakuna_matata.wav', rate, corrupted_with_out_none)


if __name__ == '__main__':
    main()

#corrupted_with_none
