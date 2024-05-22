###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

import numpy as np
from JQCLib.Classes.JMStateVectorSimClass import JMStateVectorSim

#Sample a pair of states and calculate Fidelity
def Fidelity(sim, param_low = 0, param_high = 2*np.pi):

    u1 = sim.Statevector(np.random.uniform(param_low, param_high, sim.paramCount))
    u2 = sim.Statevector(np.random.uniform(param_low, param_high, sim.paramCount))
        
    return np.abs(u1.conj().T @ u2).item()**2

#Sample list of Fidelities using JMStateVectorSim
def SampleFidelity(circuit, numSamples):

    #Create JMStateVectorSim from circuit data
    sim = JMStateVectorSim.FromArray(circuit)

    #Collect samples
    return [Fidelity(sim) for _ in range(numSamples)]

#Integration of Haar Probability Density Function
#PHaar(F) = (N − 1)(1 − F)^N−2
def f(F,N):
    return -1 * (1-F)**(N-1)

#Integral over the range of the given bin
def binProb(F1,F2,N):
    return f(F2,N)-f(F1,N)

#Compute probability distribution for a given number of bins
def HaarProb(numBins, numQubits):
    N = 2**numQubits
    return np.array([binProb(numBins[i],numBins[i+1],N) for i in range(len(numBins)-1) ])   

#Calculate Kullback–Leibler divergence from a sampled distribution
def DKLDIST(dist, numBins, numQubits):
   
    hist = np.histogram(dist, bins=numBins, range=(0,1), density=False)

    P = np.divide(hist[0],len(dist))
    Q = HaarProb(hist[1], numQubits)
    
    filter = np.where(P>0)

    P = P[filter]
    Q = Q[filter]

    return np.sum(P * np.log(P / Q))

#Sample distribution and calculate Kullback–Leibler divergence
def DKL(circuit, numBins, numSamples):
    
    #Collect samples
    dist = SampleFidelity(circuit, numSamples)

    #Calc number of qubits in circuit
    numQubits = np.array(circuit).shape[0]

    #Return DKL from samples
    return DKLDIST(dist, numBins, numQubits)


#Compute 3 metrics from a list of sampled Fidelities for a circuit:
# -First Frame potential
# -Second Frame potential
# -Kullback–Leibler divergence
def DKLFP(collected, numBins, qubits):

    ft1 = np.mean(collected)
    ft2 = np.mean(np.square(collected))
    
    return ft1, ft2, DKLDIST(collected, numBins, qubits)

