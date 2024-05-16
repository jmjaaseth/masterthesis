import numpy as np
from JQCLib.Preprocessing.Utils import ToQiskit
from JQCLib.Metrics.MeyerHilbert import HilbertSchmidtNorm, MeyerWallach

class MetricWrapper:

    #Num samples for calculating Hilbert-Schmidt norm / Meyer Wallach
    metricSamples = 100

    #Dictionary of Metrics that must be calculated and corresponding lambdas
    MustContain = {
        "Number of gates":      lambda tr,qc: np.count_nonzero(tr.CircuitData != 0),
        "Parameterized gates":  lambda tr,qc: qc.num_parameters,
        "Non-local gates":      lambda tr,qc: qc.num_nonlocal_gates(),
        "Training time":        lambda tr,qc: tr.Executiontime,
        "Training iterations":  lambda tr,qc: len(tr.trainingLoss),
        "CrossEntropy":         lambda tr,qc: tr.CrossEntropy,
        "Accuracy":             lambda tr,qc: tr.Accuracy,
        "Hilbert-Schmidt norm": lambda tr,qc: HilbertSchmidtNorm(qc, MetricWrapper.metricSamples),
        "Meyer Wallach":        lambda tr,qc: MeyerWallach(qc, MetricWrapper.metricSamples)

        #Add more metrics here
        #....
    }
    
    def __init__(self, Metrics):
        self.Metrics = Metrics

    def Validate(self, TrainResult):
        
        if(self.__isComplete()):
            return
        else:
            qiskitCircuit = ToQiskit(TrainResult.CircuitData)
            for mName in MetricWrapper.MustContain:
                if not mName in self.Metrics:
                    self.Metrics[mName] = MetricWrapper.MustContain[mName](TrainResult, qiskitCircuit)        

    def __isComplete(self):
        return np.all(np.array([mName in self.Metrics for mName in MetricWrapper.MustContain]))

    def show(self):        
        return self.Metrics