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

debug = True

def show_info(mid):
    print("Basic Info..")
    for i, track in enumerate(mid.tracks): 
        print('Track {}: {}'.format(i, track.name))
    print("Type: {}".format(mid.type))
    print("Length in seconds: {}".format(mid.length))


mid = MidiFile('midi_partitures/el_aguacate.mid')

show_info(mid)

for msg in mid.play():
    print(msg) # type, channel, note, velocity, time