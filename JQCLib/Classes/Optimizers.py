###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import numpy as np

class SimpleOptimizer:

    def __init__(self, weights, lr = .0001):        
        self.weights = weights        
        self.lr = lr

    def step(self, grads):
        self.weights -= self.lr * grads

    def getWeights(self):
        return self.weights.copy()

class Adam:
        
    def __init__(self, weights, eta=0.5, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m, self.v = 0, 0        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 1
        self.weights = weights 

    def step(self, grads):

        #Momentum beta 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        #RMS beta 2
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        #Correction
        m_corr = self.m/(1 - self.beta1**self.t)
        v_corr = self.v/(1 - self.beta2**self.t)

        self.t+=1

        #Update
        upd = self.eta*(m_corr/(np.sqrt(v_corr)+self.epsilon))

        self.weights -= upd        

    def getWeights(self):
        return self.weights.copy()