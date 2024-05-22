###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

#Import Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

#Other
import pickle
import numpy as np
from datetime import datetime

#Import thesis libs
from JQCLib.Metrics.QiskitMetrics import HilbertSchmidtNorm, MeyerWallach

class QCSampler:

    #Number of Unary gates in setup
    unarygates = 4  #:  (I = 0)  Gates: H, RX, RY, RZ

    @classmethod
    def Example1(cls):

        #Encoded circuitdata
        data = np.matrix([2,5,4,2,0,4]).reshape(2,3)

        return cls(data)
    
    @classmethod
    def Example2(cls):

        #Encoded circuitdata
        data = np.matrix([1,5,2,0,0,0]).reshape(2,3)

        return cls(data)

    @classmethod
    def Load(cls, fromFile):
        # Example:
        #   Data        hasMetrics  metrics hasScores   scores  trainTime   objValues  
        #   [np.array,   True,      .1, .1, True        .6,.55  10          np.array]

        #Create
        m = cls(np.array(fromFile[0]))

        #Metrics
        if(fromFile[1]):            
            m.SetMetrics(fromFile[2],fromFile[3])

        #Scores
        if(fromFile[4]):             
            m.SetScore(fromFile[5],fromFile[6], fromFile[7],fromFile[8])

        return m

    @classmethod
    def Sample(cls, Qubits, MaxLayers, CXProb = .3, Gateprobs = None):
        
        #Insert CX gates into a layer
        def CXGates():

            qbl = np.arange(Qubits).tolist()    #List of qubit indexes
            ent = []                            #List of encoded CX gates
            
            #Max possible number of CX gates in a layer
            for i in range(Qubits//2):

                #Insert CX gate with probability CXProb
                if(np.random.uniform(0,1) < CXProb):
                    f = qbl.pop(np.random.randint(len(qbl)))    #CX control qubit
                    t = qbl.pop(np.random.randint(len(qbl)))    #CX target qubit
                    ent.append((f,t))                           #Add to list

            #Return list of CX gates for this layer
            return ent

        #A random layer
        def RandLayer():

            #Sample unary gates using probability distribution
            if(Gateprobs == None):  #Uniform
                l = np.random.randint(0,QCSampler.unarygates+1,Qubits)
            else:                   #Distribution was given as paramater
                l = np.random.choice(QCSampler.unarygates+1, Qubits, p=Gateprobs)

            #Entanglement using CX gates
            for e in CXGates():                
                l[e[0]] = e[1] + QCSampler.unarygates + (0 if e[0] < e[1] else 1)
                l[e[1]] = 0

            return l
        
        #How many layers
        layers = np.random.randint(1, MaxLayers+1)        
        
        #Empty data
        data = np.zeros((Qubits,layers),dtype=int)

        #Sample layers
        for i in range(layers):            
            data[:,i] = RandLayer()

        #Create circuit from matrix representation
        return cls(data)   
    
    def __init__(self, cd):

        #Store the matrix representation of the circuit
        self.data = cd
        
        #Get shape of data
        shp = self.data.shape

        #Defaults
        self.hasMetrics = False
        self.hasScores = False
        self.HilbertSchmidtNorm = 0
        self.MeyerWallach = 0
        self.Experimental = 0
        self.TrainingScore = 0
        self.ValidationScore = 0
        self.trainTime = 0
        self.objValues = []

        #Extract from shape
        self.num_qubits = shp[0]
        self.num_layers = shp[1]        
        
        #Vector for encoding the parameters
        self.ip = ParameterVector('Input', np.count_nonzero((self.data == 2) | (self.data == 3) | (self.data == 4)))        

        #Create cirquit
        self.qc = QuantumCircuit(self.num_qubits)

        #Keep track of parameterized gates added
        rotCount = 0
        
        #Add gates
        for (depth,qubit) in [(i,j) for i in range(self.num_layers) for j in range(self.num_qubits)]:
            match self.data[qubit,depth]:
                case 1:
                    self.qc.h(qubit)
                case 2:            
                    self.qc.rx(self.ip[rotCount], qubit)
                    rotCount+=1            
                case 3:            
                    self.qc.ry(self.ip[rotCount], qubit)
                    rotCount+=1
                case 4:            
                    self.qc.rz(self.ip[rotCount], qubit)
                    rotCount+=1
                case _ if self.data[qubit,depth] > (QCSampler.unarygates):
                    t = self.data[qubit,depth] - QCSampler.unarygates - 1
                    
                    self.qc.cx(qubit,t if t < qubit else t+1)
                case _:
                    pass
            if(qubit == (self.num_qubits-1)):
                self.qc.barrier()

    def Qbits(self):
        return self.num_qubits

    def Circuit(self):
        return self.qc
    
    def Parameterize(self, input = None):
        if(input == None):
            input = np.random.uniform(-np.pi, np.pi, self.qc.num_parameters)        
        return self.qc.bind_parameters({self.ip: input})
    
    def SetMetrics(self, HSN, MW):
        self.hasMetrics = True
        self.HilbertSchmidtNorm = HSN
        self.MeyerWallach = MW
        self.Experimental = self.MeyerWallach - self.HilbertSchmidtNorm

    def SetScore(self, Train, Val, tTime, objVals):
        self.hasScores = True
        self.TrainingScore = Train
        self.ValidationScore = Val
        self.trainTime = tTime
        self.objValues = objVals

    def Erase(self):
        self.hasScores = False
        self.TrainingScore = 0
        self.ValidationScore = 0
        self.trainTime = 0
        self.objValues = 0   


    def Metric(self):

        metrics = []
        metrics.append(self.qc.depth())                     #Circuit depth
        metrics.append(np.count_nonzero(self.data != 0))    #Number of gates
        metrics.append(self.qc.num_parameters)              #Parameterized gates
        metrics.append(self.qc.num_nonlocal_gates())        #Non-local gates
        metrics.append(self.HilbertSchmidtNorm)             #Hilbert-Schmidt norm
        metrics.append(self.MeyerWallach)                   #Meyer Wallach
        metrics.append(self.Experimental)                   #Experimental

        return metrics
    
    def Save(self):
        return [self.data,self.hasMetrics, self.HilbertSchmidtNorm, self.MeyerWallach, self.hasScores, self.TrainingScore,self.ValidationScore, self.trainTime,self.objValues]
    
    def Info(self):

        '''
        Some more things that we could add:
            Total gate cost
            other???
        '''
        #Number of gates
        total_gates = np.count_nonzero(self.data != 0)
        nonloc_gates = self.qc.num_nonlocal_gates()
        param_gates = self.qc.num_parameters
        local_gates = total_gates - nonloc_gates - param_gates
        print("Circuit information:")
        print("--------------------------------------")
        print(" Number of qubits                :",self.qc.num_qubits)
        print(" Circuit depth                   :",self.qc.depth())
        print(" Number of gates                 :",total_gates)
        print("   -local gates                  :",local_gates)
        print("   -non-local gates              :",nonloc_gates)
        print("   -parameterized gates          :",param_gates)
        print("Has metrics                      :",self.hasMetrics)
        print("   -Hilbert-Schmidt norm         :",self.HilbertSchmidtNorm)
        print("   -Meyer Wallach                :",self.MeyerWallach)
        print("   -Experimental                 :",self.Experimental)
        print("Has scores                       :",self.hasScores)
        print("   -on training data             :",self.TrainingScore)
        print("   -on validation data           :",self.ValidationScore)
        




class Modeldata:

    @classmethod
    def Load(cls, file):

        #Unpickling
        with open(file, "rb") as fp:
            data = pickle.load(fp)        

        #Create empty list
        models = []

        for d in data[1]:
            models.append(QCSampler.Load(d))

        return cls(data[0],models)

    @classmethod
    def Sample(cls, numQubits, numModels, maxDepth, gateProbs, cxProb, calcMetrics = False, metricSamples = 100):

        #Create empty list
        models = []

        #Sample        
        while(len(models) < numModels):

            #Sample a circuit
            m = QCSampler.Sample(numQubits, maxDepth, cxProb, gateProbs)

            #guarantee a trainable circuit (must have trainable gates)
            if(m.qc.num_parameters < 1):
                continue

            if(calcMetrics):
                m1 = HilbertSchmidtNorm(m, metricSamples)
                m2 = MeyerWallach(m, metricSamples)

                #Store results in object                
                m.SetMetrics(m1,m2)
            
            models.append(m)            
            name = cls.getName(numQubits,maxDepth,cxProb)

        return cls(name,models)


    @classmethod
    def getName(cls, numQubits, numModels, cxProb):
        tm = int(datetime.timestamp(datetime.now()))
        return str(numQubits)+ '_' + str(numModels) + '_' + str(cxProb) + '_' + str(tm) + '.pkl'

    def __init__(self, name, models): 

        self.Name = name
        self.Models = models 

    def __len__(self):
        return len(self.Models)

    def __iter__(self):
        self.a = 0
        return self

    def __next__(self):
        if self.a < len(self.Models):            
            self.a += 1
            return self.Models[self.a-1]
        else:
            raise StopIteration        

    def hasMetrics(self):        
        for m in self.Models:
            if(not m.hasMetrics):
                return False
        return True

    def hasScores(self):        
        for m in self.Models:
            if(not m.hasScores):
                return False
        return True

    def Info(self):
        print("Model list '",self.Name,"' has ",len(self.Models)," models",sep='')        

    def ModelInfo(self,idx):        
        print(self.Models[idx].Info())

    def Replace(self, idx, model):        
        self.Models[idx] = model

    def Erase(self):
        #Erase scores
        for m in self.Models:        
            m.Erase()
        #Update name        
        #self.Name = Modeldata.getName(numQubits,maxDepth,cxProb)

    def reCalcMetrics(self,samples):
    
        for m in self.Models:
            
            m1 = HilbertSchmidtNorm(m,samples)
            m2 = MeyerWallach(m,samples)

            #Store results in object                
            m.SetMetrics(m1,m2)
      
    
    def Save(self, folder):        
        
        l = []
        for m in self.Models:
            l.append(m.Save())        

        filename = folder + '/' + self.Name
        with open(filename, "wb") as fp:   #Pickling
            pickle.dump([self.Name, l], fp)