import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
#print('raynest')
from corner import corner

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

    def log_likelihood(self,x):
        logl=  -1/2 * (np.sum((self.y - (x["a"] *self.x + x["b"]))**2) )  / (self.errory**2) + np.log(1/(np.sqrt(2*np.pi)*self.errory))
        return logl



x=np.linspace(0,5,10)
y=np.zeros(len(x))
a_val=2
b_val=1
sig_y=0.05
from numpy import random
for i in range(len(x)):
    mean=a_val*x[i]+b_val
    s = np.random.normal(mean, sig_y, 1)
    y[i]=s

#print(x,y)
#y= b_val + a_val * x +sig_y * np.random.rand(0 ,1, x.size)

mymodel= Line(x,y,sig_y)
nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000)
nest.run(corner = True)
post = nest.posterior_samples.ravel()

samples = np.column_stack([post[lab] for lab in mymodel.names])
fig = corner(samples, labels = ['$a$','$b$'], truths=[a_val,b_val])
fig.savefig('joint_posterior.pdf', bbox_inches = 'tight')


