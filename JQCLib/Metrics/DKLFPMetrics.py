import numpy as np

#Integration of Haar Probability Density Function
#PHaar(F) = (N − 1)(1 − F)^N−2
def f(F,N):
    return -1 * (1-F)**(N-1)

#Integral over the range of the given bin
def binProb(F1,F2,N):
    return f(F2,N)-f(F1,N)

#Compute probability distribution Q
def HaarProb(bins, qubits):
    N = 2**qubits
    return np.array([binProb(bins[i],bins[i+1],N) for i in range(len(bins)-1) ])   

#Kullback–Leibler divergence
def DKL(dist, BINS, qubits):
   
    hist = np.histogram(dist, bins=BINS, range=(0,1), density=False)

    P = np.divide(hist[0],len(dist))
    Q = HaarProb(hist[1], qubits)
    
    filter = np.where(P>0)

    P = P[filter]
    Q = Q[filter]

    return np.sum(P * np.log(P / Q))

#Compute 3 metrics from a list of sampled Fidelities for a circuit:
# -First Frame potential
# -Second Frame potential
# -Kullback–Leibler divergence
def DKLFP(collected, numBins, qubits):

    ft1 = np.mean(collected)
    ft2 = np.mean(np.square(collected))
    
    return ft1, ft2, DKL(collected ,numBins, qubits) 