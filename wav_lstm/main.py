#! /usr/bin/env python
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def graph_data(): 
    rate, data = wavfile.read('songs/las_mananitas.wav')
    for i in range(data.shape[1]):
        graph(rate, data[i:i])
        show_info("mananitas", data[i:i])

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
    graph_data()