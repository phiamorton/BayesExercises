import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
#print('raynest')
from corner import corner


class Line(raynest.model.Model):

    def __init__(self,x,y,sig_y, sig_x):
        super(Line,self).__init__()
        self.x=x
        self.y=y
        self.errorx=sig_x
        self.errory= sig_y
        
        self.names= ['a', 'b']
        self.bounds =[ [0,3], [0,3]]

        for i, xi in enumerate(self.x):
            self.names.append('x{0}'.format(i))
            self.bounds.append([xi-5*self.errorx,xi+5*self.errorx])
        
    def log_prior(self,x):
        logp=super(Line,self).log_prior(x)
        if np.isfinite(logp):
            #some prior
            return 0.
        else:
            return -np.inf

    def log_likelihood(self,x):
        logl1=-len(self.y)*np.log((np.sqrt(2*np.pi)*self.errory))
        logl2=-len(self.x)*np.log((np.sqrt(2*np.pi)*self.errorx))
        for i, xi in enumerate(self.x):

            x_s = x["x{0}".format(i)]
            logl1 += -1/2 * ((self.y[i] - (x["a"] * x_s  + x["b"]))**2)   / (self.errory**2) #+ np.log(1/(np.sqrt(2*np.pi)*self.errory)) 
            logl2 += -1/2 * ((xi - x_s )**2)  / (self.errorx**2) #+ np.log(1/(np.sqrt(2*np.pi)*self.errorx)) 
    
        logl = logl1 + logl2
        return logl


x_true=np.linspace(0,5,10)
y=np.zeros(len(x_true))
x=np.zeros(len(x_true))
a_val=2
b_val=1
sig_y=0.05
sig_x=0.1
from numpy import random
for i in range(len(x)):
    mean=a_val*x_true[i]+b_val
    s = np.random.normal(mean, sig_y, 1)
    y[i]=s
    x[i]=np.random.normal(x_true[i],sig_x)
    
#print(x,y)
#y= b_val + a_val * x +sig_y * np.random.rand(0 ,1, x.size)

mymodel= Line(x,y,sig_y, sig_x)
nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000)
nest.run(corner = True)
post = nest.posterior_samples.ravel()

samples = np.column_stack([post[lab] for lab in mymodel.names])
fig = corner(samples, labels = ['$a$','$b$'], truths=[a_val,b_val])
fig.savefig('joint_posterior.pdf', bbox_inches = 'tight')


