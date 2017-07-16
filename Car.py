import numpy as np
from enum import Enum

class CarState(Enum):
    DETECTED = 1
    TRACKED = 2
    UNTRACKED = 3

'''
state machine:
UNTRACKED -> DETECTED: any detection
DETECTED -> TRACKED:    >=M detections in N frames
DETECTED -> UNTRACKED:   <M detections in N frames
TRACKED -> DETECTED:  age > X
'''
class Car():
    def __init__(self):
        self.centroid = (0,0)
        self.w = 0
        self.h = 0
        self.id = 0
        self.age = 0 # age of last assigned detection
        self.M = 0
        self.N = 0
        self.state = CarState.UNTRACKED
        
    def transition(self, Det_thresh=3, M_thresh=3, N_thresh=5, age_thresh=5):
        self.age += 1
        if self.state==CarState.UNTRACKED and self.M>=Det_thresh:
            self.state = CarState.DETECTED
            self.age, self.M, self.N = 0, 1, 1
        elif self.state==CarState.DETECTED:
            self.N += 1
            if self.N==N_thresh and self.M>=M_thresh:
                self.state = CarState.TRACKED
                self.age, self.M, self.N = 0, 0, 0
            elif self.N==N_thresh and self.M<M_thresh:
                self.state = CarState.UNTRACKED
                self.age, self.M, self.N = 0, 0, 0
        elif self.state==CarState.TRACKED and self.age>age_thresh:
            self.state = CarState.DETECTED
            self.age, self.M, self.N = 0, 0, 0
            
