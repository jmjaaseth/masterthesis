###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import numpy as np

#Quantum Neural Network class
class jQNN:
        
    #Constructor
    def __init__(self, estimator, probMapper, randomweights = False):

        #Set functions
        self.estimator = estimator
        self.probMapper = probMapper

        #Weight initialization
        if(randomweights):
            self.weights = np.random.uniform(-np.pi,np.pi,estimator.paramCount)        
        else:
            self.weights = np.zeros(estimator.paramCount)        

    def CircuitData(self):
        return self.estimator.CircuitData

    #Show underlying circuit.
    def show(self, Reversed = False):
        self.estimator.show(Reversed)

    #Model's forward function
    def forward(self, features, weights):

        #Map distribution to new distribution of length = classes
        return self.probMapper(self.estimator.forward(features,weights))

    #Predict with model
    def predict(self, features, weights):
        return np.argmax(self.forward(features,weights))  
    