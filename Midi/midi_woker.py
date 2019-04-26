#Taken from http://blog.espol.edu.ec/estg1003/midi-un-instrumento/

import mido as md
midi_songs_path = 'Midi_songs/'

# Ingreso
archivo_mid = midi_songs_path + 'el_aguacate.mid'
archivo_txt = 'el_aguacatemidi.txt'

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
archivo.close()



