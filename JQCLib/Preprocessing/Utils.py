###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import pandas as pd
import numpy as np
import pickle
import os.path

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from JQCLib.Utils.Web import WebDecode

def createDemoData(file):

    #Metric labels
    labels = ["Id", "Number of gates", "Parameterized gates",  "Non-local gates",  "Training time",  "Training iterations", "MSELoss", "Accuracy", "Hilbert-Schmidt norm", "Meyer Wallach"]

    #Data
    data = np.array([
        [0, 14, 1,  5, 245, 21, .20, .89, .29, .50],
        [1, 13, 9,  6, 87,  14, .22, .88, .28, .45],
        [2, 12, 7,  7, 103, 18, .24, .87, .27, .40],
        [3, 11, 13, 8, 129, 9, .26, .86, .26, .35],
        [4, 10, 1,  9, 64,  11, .28, .85, .25, .30]
    ])

    #Create dataframe
    df = pd.DataFrame(data, columns=labels)

    #Cast applicable columns to INT and set Index as the MySql index
    df = df.astype({"Id": int, "Number of gates": int, "Parameterized gates": int, "Non-local gates": int, "Training iterations": int})    
    df = df.set_index('Id')
    
    #Save file
    df.to_csv(file)

def thesisData(file):    
    return pd.read_csv(file, index_col="Id")    

#Convert array to Qiskit Circuit
def ToQiskit(data, insert_barriers=True):
    
    #Get shape of data
    shp = data.shape

    #Extract from shape
    num_qubits = shp[0]
    num_layers = shp[1]
    
    #Number of Unary gates in setup
    unarygates = 4  #:  (I = 0)  Gates: H, RX, RY, RZ
    
    #Vector for encoding the parameters
    ip = ParameterVector('Input', np.count_nonzero((data == 2) | (data == 3) | (data == 4)))        

    #Create cirquit
    qc = QuantumCircuit(num_qubits)

    #Keep track of parameterized gates added
    rotCount = 0
    
    #Add gates
    for (depth,qubit) in [(i,j) for i in range(num_layers) for j in range(num_qubits)]:
        match data[qubit,depth]:
            case 1:
                qc.h(qubit)
            case 2:            
                qc.rx(ip[rotCount], qubit)
                rotCount+=1            
            case 3:            
                qc.ry(ip[rotCount], qubit)
                rotCount+=1
            case 4:            
                qc.rz(ip[rotCount], qubit)
                rotCount+=1
            case _ if data[qubit,depth] > (unarygates):
                t = data[qubit,depth] - unarygates - 1
                
                qc.cx(qubit,t if t < qubit else t+1)
            case _:
                pass
        if(qubit == (num_qubits-1) and insert_barriers):
            qc.barrier()
    
    #Return circuit
    return qc

def readWebCircuits(downloadedFile):

    trainedModels = {}
    #Read file and remove the new line characters
    with open(downloadedFile) as f:
        lines = [line.rstrip().split(";") for line in f]

    #Create a dictionary from stripped lines
    for key, value in lines:
        trainedModels[int(key)] = pickle.loads(WebDecode(value))

    return trainedModels

def saveMetrics(file, dictionary):

    #Save pickled data
    with open(file, 'wb') as pfile: 

        # A new file will be created 
        pickle.dump(dictionary, pfile)

def loadMetrics(file):

    if not os.path.isfile(file):
        return {}

    with open(file, 'rb') as infile:
        loaded = pickle.load(infile)

    return loaded

def loadMoreMetrics(directory):

    loaded = {}
    for file in os.listdir(directory):        
        with open(directory + "\\" + file, 'rb') as infile:            
            loaded.update(pickle.load(infile))
    return loaded