import numpy as np
import qutip

#Import Qiskit
from qiskit import Aer, execute, quantum_info

def compute_Q_ptrace(ket, N, verbose = False):

    """Computes Meyer-Wallach measure using alternative interpretation, i.e. as
    an average over the entanglements of each qubit with the rest of the system
    (see https://arxiv.org/pdf/quant-ph/0305094.pdf).
   
    Args:
    =====
    ket : numpy.ndarray or list
        Vector of amplitudes in 2**N dimensions
    N : int
        Number of qubits
​
    Returns:
    ========
    Q : float
        Q value for input ket
    """

    #Verbose printing
    def xprint(*args, **kwargs):
        if(verbose):
            print( "Verbose: "+" ".join(map(str,args)), **kwargs)   

    #Ket
    ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()
    xprint('KET=  ', ket)

    entanglement_sum = 0
    for k in range(N):
        xprint('value of n', k, 'PTrace: ',ket.ptrace([k])**2 )
        rho_k_sq = ket.ptrace([k])**2
        entanglement_sum += rho_k_sq.tr()  
   
    Q = 2*(1 - (1/N)*entanglement_sum)
    return Q

def random_unitary(N):
    """
        Return a Haar distributed random unitary from U(N)
    """

    Z = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
    [Q, R] = np.linalg.qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)

def haar_integral(num_qubits, samples):
    """
        Return calculation of Haar Integral for a specified number of samples.
    """

    N = 2**num_qubits
    randunit_density = np.zeros((N, N), dtype=complex)
    
    zero_state = np.zeros(N, dtype=complex)
    zero_state[0] = 1
    
    for _ in range(samples):
        A = np.matmul(zero_state, random_unitary(N)).reshape(-1,1)
        randunit_density += np.kron(A, A.conj().T) 

    randunit_density/=samples

    return randunit_density   


def pqc_integral_test(circuit, samples):
    """
        Return calculation of Integral for a PQC over the uniformly sampled 
        the parameters θ for the specified number of samples.
    """
    N = circuit.num_qubits
    randunit_density = np.zeros((2**N, 2**N), dtype=complex)

    for _ in range(samples):

        ansatz = Parameterize(circuit)
        result = execute(ansatz, backend=Aer.get_backend('statevector_simulator')).result()
        
        sv = result.get_statevector(ansatz, decimals=5)
        U = np.array(sv).reshape(-1,1)

        randunit_density += np.kron(U, U.conj().T)

    return randunit_density/samples



def Parameterize(qc):    
    return qc.bind_parameters(np.random.uniform(-np.pi, np.pi, qc.num_parameters))


def HilbertSchmidtNorm(circuit, samples = 2048):    
    return np.linalg.norm(haar_integral(circuit.num_qubits, samples) - pqc_integral_test(circuit, samples))

def MeyerWallach(circuit, sample=2048):
    
    """
        Returns the meyer-wallach entanglement measure for the given circuit. 
    """

    res = np.zeros(sample, dtype=complex)
    N = circuit.num_qubits

    for i in range(sample):
        
        ansatz = Parameterize(circuit)
        result = execute(ansatz, backend=Aer.get_backend('statevector_simulator')).result()
        U = result.get_statevector(ansatz, decimals=5)
        entropy = 0
        qb = list(range(N))

        for j in range(N):
            dens = quantum_info.partial_trace(U, qb[:j]+qb[j+1:]).data
            trace = np.trace(dens**2)
            entropy += trace

        entropy /= N
        res[i] = 1 - entropy
    
    return 2*np.sum(res).real/sample