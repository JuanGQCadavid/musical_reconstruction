#Imports
import mido as md
import sys 
class MidiApi():

    
    def __init__(self, midi_partiture_path,debug):
        self.midi_partiture_path = midi_partiture_path
        self.midi_partiture_name = midi_partiture_path.split('/')[-1]
        self.midi_partiture_path_relative =  midi_partiture_path[0:len(midi_partiture_path) - len(self.midi_partiture_name)]
        self.midi_partiture = md.MidiFile(midi_partiture_path)
        self.debug = debug
        
        self.print('Loading partiture ', self.midi_partiture_name, ' from ', self.midi_partiture_path_relative)
        return
    
    def print(self,*str_i):
        if(self.debug):
            print(str(str_i))

    def getDict(self,chanel=None):
        return

    def printStructrue(self,save_in_file=False):
        out_file = None
        if(save_in_file):
            path = self.midi_partiture_path_relative + self.midi_partiture_name.split('.')[0] + '.out'
            out_file = open(path,'w')

        for midi_msg in self.midi_partiture:
            print(midi_msg)

            if(save_in_file):
                out_file.write(str(midi_msg) + '\n')
                dict_d = midi_msg.dict()
                out_file.write(str(dict_d.keys()) + '\n')
                out_file.write(str(dict_d.values()) + '\n')
                out_file.write('\n')
        if(save_in_file):
            out_file.close()

        
