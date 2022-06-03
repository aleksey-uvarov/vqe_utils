from skopt import gp_minimize, forest_minimize, gbrt_minimize
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import TensorNetwork
from math import pi
import TNOptimize
import hamiltonians
from numpy import array, float64, ndarray
# from qiskit_qcgpu_provider import QCGPUProvider

# Provider = QCGPUProvider()
Provider = None


from typing import Callable, List

sv_backend = Aer.get_backend("statevector_simulator")
qasm_backend = Aer.get_backend("qasm_simulator")
# gpu_backend = Provider.get_backend("statevector_simulator")
gpu_backend = sv_backend
# sv_backend = Aer.get_backend("statevector_simulator")
# noisy_backend = Aer.get_backend(!!!)


def measure_ham_2(circ, ham_dict=None, explicit_H=None, shots=1000, backend="cpu", **kwargs):
    '''Measures the expected value of a Hamiltonian passed either as
    ham_dict (uses QASM backend) or as explH (uses statevector backend)
    '''
    if ham_dict is not None:
        E = 0
        for key, value in ham_dict.items():
            E += value * get_en_2(circ, key, shots=shots, **kwargs)
        return E
    elif explicit_H is not None:
        if backend == "gpu":
            print("GPU support is disabled for now")
            job = execute(circ, gpu_backend) #Maybe no **kwargs here
        elif backend == "cpu":
            job = execute(circ, sv_backend, **kwargs) #Maybe no **kwargs here
        else:
            raise ValueError("gpu or cpu")
        result = job.result()
        state = result.get_statevector(circ)
        state = array(state).reshape((len(state), 1))
        E = (state.T.conj() @ explicit_H @ state)[0,0].real
        return E
    else:
        raise TypeError('pass a dictionary or an explicit Hamiltonian matrix')
        

def get_en_2(circuit_in, ham_string, shots=1000, **kwargs):
    """Assumes that there are few"""
    q = circuit_in.qregs[0]
    c = circuit_in.cregs[0]
    circ = QuantumCircuit(q, c)
    
    circ.data = []
    
    if ham_string == 'I' * len(q):
        return 1
    
    for i in range(len(ham_string)):
        if ham_string[i] == 'X':
            circ.h(q[i])
        if ham_string[i] == 'Y':
            circ.sdg(q[i])
            circ.h(q[i])
        if ham_string[i] != 'I':
            circ.measure(q[i], c[i])
    circ = circuit_in + circ
    job = execute(circ, qasm_backend, shots=shots, **kwargs)
    result = job.result()
    answer = result.get_counts()
    expected_en = 0
    for key in answer.keys():
        expected_en += answer[key] * (-1)**key.count('1') / shots
    return expected_en
    

def build_objective_function(TN: TensorNetwork.TensorNetwork,
                             explicit_h: ndarray = None,
                             ham_dict: dict = None,
                             shots=1000,
                             backend="cpu", **kwargs) -> Callable[[List[float]], float]:
    '''
    Takes the tensor network, Hamiltonian and returns a function R^k -> R. 
    Maybe pass actual qiskit backends, not their string names
    '''
    def f(x):
        circ = TN.construct_circuit(x)
        return float64(measure_ham_2(circ, explicit_H=explicit_h,
                                     ham_dict=ham_dict, backend=backend,
                                     shots=shots, **kwargs))
    return f

def globalVQE_2(TN, ham_dict, use_explicit_H=True, n_calls=100, initial_circuit=None, verbose=True):

    def total_circ(x):
        if initial_circuit:
            return initial_circuit + TN.construct_circuit(x)
        else:
            return TN.construct_circuit(x)
            
    if use_explicit_H:
        H = hamiltonians.explicit_hamiltonian(ham_dict)
        def objective(x):
            return measure_ham_2(total_circ(x), explicit_H=H)
    else:
        def objective(x):
            return measure_ham_2(total_circ(x), ham_dict=ham_dict)

    res = gp_minimize(objective, [(0, 2 * pi)] * TN.n_params, n_calls=n_calls, verbose=verbose, x0=[0]*TN.n_params)
    return res

def any_order_VQE_2(TN: TensorNetwork, params_order: list,
                    init_vals=None,
                    ham_dict=None, explicit_H=None,
                    initial_circuit=None,
                    n_calls=100, verbose=True):
    '''
    Performs VQE by optimizing the tensor network in the order
    supplied in params_order.
    '''
    if init_vals:
        vals = [u for u in init_vals]
    else:
        vals = [0] * TN.n_params
    
    for free_parameters in params_order:
        print('Optimizing parameters ', free_parameters, end=' ... ')
        f = restrained_objective_2(TN, free_parameters, vals,
                                   explicit_H=explicit_H, ham_dict=ham_dict,
                                   initial_circuit=initial_circuit)
        suggested_point = [vals[i] for i in free_parameters]

        ## Supposedly the unitary exp(i F \theta) yields a pi-periodic
        ## cost function if F is a Pauli operator or any such that F**2 = 1
        ## Which is sadly not always the case
        res = gp_minimize(f, [(0,  2 * pi)] * len(free_parameters), n_calls=n_calls,
                          x0=suggested_point, verbose=verbose)

        
        #print(res.x)
        for i, n in enumerate(free_parameters):
            vals[n] = res.x[i]
        print('E = {0:0.4f}'.format(res.fun))
        #print(['{0:0.6f}'.format(v) for v in vals])
        #print(measure_ham_2(TN.construct_circuit(vals), explicit_H=explicit_H))

    return res.fun, vals
        

def restrained_objective_2(TN, free_parameters, default_vals, ham_dict=None, explicit_H=None, initial_circuit=None):
    '''Makes an objective function good for minimization. Locks most
    parameters, while leaving those listed in free_parameters as free
    '''

    def f(x):
        assert(len(x) == len(free_parameters)), 'Free parameters qty mismatch!'
        params = [u for u in default_vals]  ## this is probably bad
        for i, n in enumerate(free_parameters):
            params[n] = x[i]
        #print(params)
        circ = TN.construct_circuit(params)
        if initial_circuit:
            circ = initial_circuit + circ
        return measure_ham_2(circ, ham_dict=ham_dict, explicit_H=explicit_H)
    return f