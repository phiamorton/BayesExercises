import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
#print('raynest')

class Line(raynest.model.Model):

    def __init__(self,x,y,sig_y):
        super(Line,self).__init__()
        self.x=x
        self.y=y
        self.errory= sig_y
        self.names= ['a', 'b']
        self.bounds =[ [0,3], [0,3]]

    def log_prior(self,x):
        logp=super(Line,self).log_prior(x)
        if np.isfinite(logp):
            #some prior
            return 0.
        else:
            return -np.inf

