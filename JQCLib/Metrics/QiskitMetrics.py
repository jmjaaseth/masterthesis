###########################################
#      Code By Jørgen Mjaaseth, 2023      #
###########################################

#Implementation of Hilbert-Schmidt Norm & Meyer-Wallach Entanglement Measure using Qiskit

#Imports
from qiskit import Aer, execute, quantum_info
import numpy as np

#Haar distributed random unitary
def RandomUnitary(N):
    Z = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
    [Q, R] = np.linalg.qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)

#Sampled estimation of Haar Integral
def SampledHaarIntegral(numQubits, numSamples):

    N = 2**numQubits
    randunit_density = np.zeros((N, N), dtype=complex)
    
    zero_state = np.zeros(N, dtype=complex)
    zero_state[0] = 1
    
    for _ in range(numSamples):
        A = np.matmul(zero_state, RandomUnitary(N)).reshape(-1,1)
        randunit_density += np.kron(A, A.conj().T) 

    randunit_density/=numSamples

    return randunit_density   

#Estimated inegral of PQC over uniformly sampled parameters
def SampledPQCIntegral(circuit, numSamples):

    N = 2**circuit.num_qubits
    randunit_density = np.zeros((N, N), dtype=complex)

    for _ in range(numSamples):

        ansatz = circuit.Parameterize()
        result = execute(ansatz, backend=Aer.get_backend('statevector_simulator')).result()
        
        sv = result.get_statevector(ansatz, decimals=5)
        U = np.array(sv).reshape(-1,1)

        randunit_density += np.kron(U, U.conj().T)

    return randunit_density/numSamples

#Descriptor: Hilbert-Schmidt Norm
def HilbertSchmidtNorm(circuit, samples = 2048):    
    return np.linalg.norm(SampledHaarIntegral(circuit.num_qubits, samples) - SampledPQCIntegral(circuit, samples))

#Descriptor: Meyer-Wallach Entanglement Measure
def MeyerWallach(circuit, sample=2048):
    res = np.zeros(sample, dtype=complex)   

    for i in range(sample):
        
        ansatz = circuit.Parameterize()
        result = execute(ansatz, backend=Aer.get_backend('statevector_simulator')).result()
        U = result.get_statevector(ansatz, decimals=5)
        entropy = 0
        qb = list(range(circuit.num_qubits))

        for j in range(circuit.num_qubits):
            dens = quantum_info.partial_trace(U, qb[:j]+qb[j+1:]).data
            trace = np.trace(dens**2)
            entropy += trace

        entropy /= circuit.num_qubits
        res[i] = 1 - entropy
    
    return 2*np.sum(res).real/sample