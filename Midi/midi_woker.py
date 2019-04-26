import mido as md
from MidiApi import MidiApi as midiApi


midi_songs_path = 'midi_partitures/'
song_name = 'mary' 

api = midiApi(midi_songs_path+song_name + '.mid',debug=True)
api.printStructrue(save_in_file=True)


'''
# Ingreso
archivo_mid = midi_songs_path + song_name+ '.mid'
archivo_txt = midi_songs_path + song_name + '.out'

#Procedimiento
partitura = md.MidiFile(archivo_mid)

#Salida
pistas = partitura.tracks
n = len(pistas)

for i in range(0,n,1):
    print(i,pistas[i])

#Archivo texto
archivo = open(archivo_txt,'w')
for dato in partitura:
    archivo.write(str(dato) + '\n')

    dict_d = dato.dict()
    archivo.write(str(dict_d.keys()) + '\n')
    archivo.write(str(dict_d.values()) + '\n')
    archivo.write('\n')

archivo.close()

#Taken from http://blog.espol.edu.ec/estg1003/midi-un-instrumento/
for msg in  partitura:
    
    print(type(msg))
    print(msg)

'''