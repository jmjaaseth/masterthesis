import numpy as np
from matplotlib import pyplot as plt

class TrainingResult:
        
    def __init__(self, CircuitData):

        self.CircuitData = CircuitData

        self.trainingLoss = []
        self.validationLoss = []

        self.Executiontime = 0
        self.Accuracy = 0
        self.CrossEntropy = 0

    def report(self, train, val, verbose = False):
        self.trainingLoss.append(train)
        self.validationLoss.append(val)
        if(verbose):
            print(f"Epochs: {len(self.trainingLoss)} : {train} {val}")

    def execTime(self, t):
        self.Executiontime = t
    
    def bestIndex(self):
        return np.argmin(np.array(self.validationLoss))
    
    def setScores(self, Accuracy, CrossEntropy):
        self.Accuracy = Accuracy
        self.CrossEntropy = CrossEntropy
    
    def plot(self):

        plt.plot(range(len(self.trainingLoss)),self.trainingLoss, label='Training loss')
        plt.plot(range(len(self.validationLoss)),self.validationLoss, label='Validation loss')
        plt.legend()