###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import numpy as np

#Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler

#Thesis imports
from JQCLib.Classes.QCSamplerClass import QCSampler
from JQCLib.Utils.Utilities import fromRaw

#Wrapper around Qiskit's Sampler, to make it into a model 
class jQiskitSampler:

    @classmethod
    def FromArray(cls, data):
        return cls(fromRaw(data))

    def __init__(self, CircuitData):
        
        ansatz = QCSampler(CircuitData).Circuit()

        #self.sampler = Sampler(options = {'shots':100})
        self.CircuitData =      CircuitData
        self.sampler = Sampler()
        self.no_qubits = ansatz.num_qubits
        self.out_dim = 2**self.no_qubits
        self.paramCount = len(ansatz.parameters)

        #The circuit
        self.circuit = QuantumCircuit(self.no_qubits)
        self.circuit.compose(self.__getFeatureMap(), range(self.no_qubits), inplace=True)
        self.circuit.compose(ansatz, range(self.no_qubits), inplace=True)
        self.circuit.measure_all()        
        
    def __getFeatureMap(self):

        feature_map = QuantumCircuit(self.no_qubits)

        #Vector for encoding the features
        ip = ParameterVector('Features',self.no_qubits) 

        for i in range(self.no_qubits):
            feature_map.h(i)
            feature_map.rz(ip[i], i)

        return feature_map

    def __getQuasiProb(self, features, weights):

        job = self.sampler.run(self.circuit, features.tolist() + weights.tolist())
        return job.result().quasi_dists[0]
    
    def __addMissing(self, dict):
        dist = [0] * self.out_dim
        for key, val in dict.items():
            dist[key] = val
        return dist
    
    def show(self,Reversed = False):
        display(self.circuit.draw("mpl",reverse_bits = Reversed))    
    
    def forward(self, features, weights):

        #Add all values
        return self.__addMissing(self.__getQuasiProb(features, weights))