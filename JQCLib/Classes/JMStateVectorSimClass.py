###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import numpy as np
from itertools import product
import functools as ft
from JQCLib.Utils.Utilities import fromRaw


#Gates & function definitions

#Define |0>
k0 = np.array([1,0]).reshape(2,1)

#Controlled X gate
CX = np.matrix([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
])

#Rotational gates
def Rx(θ):
    return np.matrix([
        [np.cos(θ/2),           (-1j) * np.sin(θ/2)   ],
        [(-1j) * np.sin(θ/2),     np.cos(θ/2)         ]
    ])

def Ry(θ):
    return np.matrix([
        [np.cos(θ/2),       -np.sin(θ/2)   ],
        [np.sin(θ/2),       np.cos(θ/2)         ]
    ])

def Rz(θ):
    return np.matrix([
        [np.exp(-1j*θ/2),   0               ],
        [0,                 np.exp(1j*θ/2)  ]
    ])

#Unary gates
def H():
    return np.matrix([
        [1,1],
        [1,-1]
    ])*1/np.sqrt(2)

def I():
    return np.matrix([
        [1,0],
        [0,1]
    ])

def X():
    return np.matrix([
        [0,1],
        [1,0]
    ])

#CX Gate
def Cx(Skips):
    
    l1 = [np.array([1,0,0,0]).reshape(2,2)]
    l2 = [np.array([0,0,0,1]).reshape(2,2)]

    l1 += [np.eye(2)]*abs(Skips)
    l2 += [np.eye(2)]*(abs(Skips)-1)
    l2.append(np.matrix([0,1,1,0]).reshape(2,2))

    cx= ft.reduce(np.kron,l1) + ft.reduce(np.kron,l2)

    if(Skips > 0):
        return cx
    else:
        hg = ft.reduce(np.kron,[H()] + [I()]*(abs(Skips)-1) + [H()])        
        return hg @ cx @ hg
    
#Helper class to encode features as np.matrix
def FeatureEncode(input, QiskitOrder = True):

    #How many qubits
    qubits = len(input)

    if(QiskitOrder):
        input.reverse()

    #Initiate circuit with |0>
    zl = ft.reduce(np.kron,[k0]*qubits)

    #A Hadamard layer
    hl = ft.reduce(np.kron,[H()] * qubits)

    #A rotation layer    
    rl = ft.reduce(np.kron,[Rz(theta) for theta in input] )

    #Return encoded state (np.matrix)
    return rl @ hl @ zl
    
#State vector simulator
class JMStateVectorSim:

    @classmethod
    def FromArray(cls, data, QiskitOrder = True):
        return cls(fromRaw(data),QiskitOrder)

    def __init__(self, CircuitData, QiskitOrder = True):

        #Properties
        self.CircuitData =      CircuitData
        self.NumQubits =        len(CircuitData)        
        self.__QiskitOrder =    QiskitOrder
        
        #Create circuit from given representation
        Circuit = self.__buildunitary(CircuitData)

        #Obsolete Qubit selector
        #self.Selector =self.__getSelector()
        
        #Measurement operator (standard base)
        self.__M =      self.__getOperator()
        
        #Complete calculation, with state included
        self.__MTRX =   lambda State, theta : Circuit(theta) @ State

    def __buildunitary(self,tc):       

        def LayerData(layer):
            
            data = []
            
            i = 0
            #Iterate over gates in layer
            while(i < len(layer)):
                
                #Current gate
                gate = layer[i]

                #Unary gates
                if(gate == 0):
                    data.append(lambda _ : I())                    
                elif(gate == 1):
                    data.append(lambda _ : H())

                #Rotation gates
                elif(gate == 2):
                    data.append(lambda theta,i=self.paramCount : Rx(theta[i]))
                    self.paramCount+=1
                elif(gate == 3):
                    data.append(lambda theta,i=self.paramCount : Ry(theta[i]))
                    self.paramCount+=1
                elif(gate == 4):
                    data.append(lambda theta,i=self.paramCount : Rz(theta[i]))
                    self.paramCount+=1

                #Binary gates
                else:
                    target = gate-5
                    if(target >= i):
                        target+=1                    
                    skips = target - i                    

                    if(skips < 0):
                        del data[skips:]
                    else:
                        i+=skips

                    if(self.__QiskitOrder):
                        skips = -skips

                    data.append(lambda _ : Cx(skips))
                i+=1    
            
            #QiskitOrder True   : Qn ⊗ ... ⊗ Q0
            #QiskitOrder False  : Q0 ⊗ ... ⊗ Qn
            if(self.__QiskitOrder):
                data.reverse()

            #Return tensor product of layer gates
            return lambda theta : ft.reduce(np.kron,[f(theta) for f in data])

        def SplitLayers(raw):
            def sublist(array,indices,length):
                sl = np.zeros(length,dtype=int)
                sl[indices] = array[indices]
                return sl.tolist()

            splittedlists = []
            l = len(raw)
            
            #Extract all unary gates
            splittedlists.append(sublist(raw,np.where(np.logical_and(raw>0, raw<=4)),l))

            #Extract binary gates, one by one
            for ind in np.where(raw>4)[0]:        
                splittedlists.append(sublist(raw,ind,l))
            
            return splittedlists

        #Learnable parameters
        self.paramCount = 0        

        #Split layers to avoid problems with multiple Cx gates
        splitted = []
        for c in tc.T:
            splitted+=SplitLayers(c)        

        #Build layer by layer
        layers = [LayerData(c) for c in splitted]

        #Return product of layers in reversed order
        return lambda theta : ft.reduce(np.dot,[f(theta) for f in reversed(layers)])

    def __getOperator(self):

        #Base we will be using: computational (Z-base)
        ZBASE = [
            np.array([1,0]).reshape(2,1),
            np.array([0,1]).reshape(2,1)
        ]
        #Return standard base measurement operator        
        return [np.matrix(ft.reduce(np.kron, p)).getH() for p in product(*np.array([ZBASE] * self.NumQubits))]

    #Pick the correct probabilities that should be summed
    def __getSelector(self):
        def op(q,p):
            return [i for i in range(2**q) if((i//2**p)%2 > 0)]
        uu = [op(self.NumQubits,i) for i in range(self.NumQubits)]
        uu.reverse()
        return np.array(uu)

    #Returns probability distribution for possible states
    def __measure(self, state, weights):

        #Compute U @ Input        
        u = self.__MTRX(state,weights)
        
        #Return list of probabilities for the possible states        
        return [(np.abs(p @ u)**2).item(0) for p in self.__M]
    
    #Test Unitary with some weights on a given input state. 
    def Statevector(self,theta, state = None):       

        #Using inital state of all zeros
        if(state is None):            

            #Define |0>
            k0 = np.array([1,0]).reshape(2,1)

            #Create state
            state = ft.reduce(np.kron,[k0]*self.NumQubits)

        #Compute and return
        return self.__MTRX(state,theta)

    def show(self, Reversed = False):
        print("Not implemented")

    def forward(self, features, weights):
        return self.__measure(features, weights)