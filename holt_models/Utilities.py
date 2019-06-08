#!/usr/bin/env python3.6
import random

class Utilities:
    
    def __init__(self,percent, intervals ,level):
        self.percent = percent
        self.intervals = intervals
        self.level = level

    def corrupt(self,data):
        if self.level == 1:
            return self.odebrecht(data)
    
    def odebrecht(self,data):
        data_len = len(data)
        data_broken = [False]*len(data)
        frec_iterator = 0
        put_none_counter = 0
        for frec_iterator in range(0,data_len):
            if frec_iterator < 500:
                continue
            if put_none_counter > 0:
                data_broken[frec_iterator] = True
                data[frec_iterator] =  0
                put_none_counter -= 1
            else:
                r = random.random()
                if r < self.percent:
                    put_none_counter = self.intervals
        
        return data,data_broken
    
    def reconstruction(self,data):
        for frec_iterator in range(0,len(data)):
            if data[frec_iterator] == None:
                data[frec_iterator] = 0
        return data
            

