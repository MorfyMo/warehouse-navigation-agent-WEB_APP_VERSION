"""
This class helps to compute the running statistics (i.e. running mean, running standard deviation).
Goal: to ease the scaling(normalization) of the reward in Agent class
*Note: we need to normalize the reward because the reward is sometimes too extreme;
    thus in order to make it easy to run in this environment, we need to normalize the reward.
"""
import numpy as np

class Running_Stats:
    def __init__(self):
        self.count=0
        self.mean=0.0
        self.M2=0.0 #sum of squares of deviations from the mean
    
    def update(self,x):
        self.count+=1
        delta=x-self.mean
        self.mean+=(x-self.mean)/self.count
        
        delta2=x-self.mean
        self.M2+=delta*delta2
    
    def get_mean(self):
        return self.mean
    
    #Note that this is the population version;
    #if we want the alternative sample version, we have "np.sqrt(self.M2/(self.count-1))"
    def get_std(self):
        return np.sqrt(self.M2/self.count) if self.count>0.0 else 1.0