from numpy import eye, array, diag, zeros, kron, complex64
from numpy.linalg import eig, eigh
from functools import reduce
from itertools import product, combinations
#from scipy.sparse import csr_matrix
import numpy as np


I = eye(2)
Z = diag([1, -1])
X = array([[0, 1], [1, 0]])
Y = array([[0, -1j], [1j, 0]])
paulis = (I, X, Y, Z)
pauli_labels = ("I", "X", "Y", "Z")


def ising_model(n_spins, J, hx):
    ham = {}
    line = 'Z' + 'Z' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = J
    line = 'X' + 'I' * (n_spins - 1)
    if hx != 0:
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = hx
    return ham

def heis_model(n_spins, J, hx):
    ham = {}
    for spin in ['X', 'Y', 'Z']:
        line = spin + spin + 'I' * (n_spins - 2)
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = J
    line = 'X' + 'I' * (n_spins - 1)
    if hx != 0:
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = hx
    return ham

def xxz_heisenberg_model(n_spins, J_x, J_z):
    ham = {}
    for spin in ['X', 'Y']:
        line = spin + spin + 'I' * (n_spins - 2)
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = J_x
    line = 'Z' + 'Z' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = J_z
    return ham


def local_fields_hamiltonian(n_qubits, local_fields):
    """Adds a local magnetic field term on each qubit. Needs 3 * n_qubits real values"""
    ham = {}
    for i in range(n_qubits):
        for j, s in enumerate(['X', 'Y', 'Z']):
            key = "I" * i + s + "I" * (n_qubits - i - 1)
            ham[key] = local_fields[j + i * 3]
    return ham


def two_d_heisenberg(n_x, n_y, J):
    """Heisenberg model on a patch of square lattice. 
    Periodic boundary conditions"""
    ham = {}
    for i in range(n_x * n_y):
        if ((i + 1) % n_y == 0):
            horizontal_pair = (i, (i - n_y + 1))
        else:
            horizontal_pair = (i, (i + 1))
        if (i + n_y >= n_x * n_y):
            vertical_pair = (i, (i + n_y) % (n_x * n_y))
        else:
            vertical_pair = (i, (i + n_y))
        for s in ["X", "Y", "Z"]:
            key_list = ["I"] * n_x * n_y
            key_list[horizontal_pair[0]] = s
            key_list[horizontal_pair[1]] = s
            key = reduce(lambda a, b: a + b, key_list)
            ham[key] = J

            key_list = ["I"] * n_x * n_y
            key_list[vertical_pair[0]] = s
            key_list[vertical_pair[1]] = s
            key = reduce(lambda a, b: a + b, key_list)
            ham[key] = J
    return ham

        
def two_d_heisenberg_with_local_fields(n_x=3, n_y=3, J=1, local_fields=None):
    ham_1 = two_d_heisenberg(n_x, n_y, J)
    ham_2 = local_fields_hamiltonian(n_x * n_y, local_fields)
    return {**ham_1, **ham_2}


def magnetic_field(n_spins, h, direction='X'):
    """
    External magnetic field acting on all spins
    :param n_spins: qty of spins
    :param direction: 'X', 'Y', or 'Z'
    :param h: field
    :return: dictionary with terms
    """

    ham = {}
    line = direction + 'I' * (n_spins - 1)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = h
    return ham


def XY_model(n_spins, gamma, g):
    ''' XY model with transverse field'''
    ham = {}
    line = 'X' + 'X' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = 0.5 * (1 + gamma)
        
    line = 'Y' + 'Y' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = 0.5 * (1 - gamma)
        
    line = 'Z' + 'I' * (n_spins - 1)
    if g != 0:
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = g
    return ham

### ---------

def explicit_hamiltonian(ham_dict):
    n_qubits = len(list(ham_dict.keys())[0])
    I = eye(2)
    X = array([[0, 1], [1, 0]])
    Y = array([[0, -1j], [1j, 0]])
    Z = diag([1, -1])
    pauli={}
    pauli['I'] = I
    pauli['X'] = X
    pauli['Y'] = Y
    pauli['Z'] = Z
    H = zeros((2**n_qubits, 2**n_qubits), dtype='complex128')
    for term, energy in ham_dict.items():
        matrices=[]
        for sym in term:
            matrices.append(pauli[sym])
        total_mat = energy * reduce(kron, matrices)
        H +=total_mat
    return H


def H_to_pauli_dict(H, reversed=True):
    """Takes a qubit Hamiltonian and returns its Pauli decomposition.
    If reversed is True, the Pauli strings are reversed. I\'m not sure if 
    the need for that stems from some bug in my code or from the endianness
    conventions of Qiskit"""
    assert(H.shape[0] == H.shape[1])
    assert(np.isclose(np.log2(H.shape[0]) % 1, 0))
    n_qubits = round(np.log2(H.shape[0]))
    labels_iterator = product(pauli_labels, repeat=n_qubits)
    matrices_iterator = product(paulis, repeat=n_qubits)
    ham_dict = {}
    for label_list, matrix_list in zip(labels_iterator, matrices_iterator):
        key = reduce(str.__add__, label_list)
        pauli_string = reduce(np.kron, matrix_list)
        value = np.trace(pauli_string @ H) / 2**(n_qubits)
        if value != 0:
            ham_dict[key] = value.real
            if not np.isclose(value.imag, 0):
                raise ValueError("Only Hermitian matrices please")
    if reversed:
        return {k[::-1]: v for k, v in ham_dict.items()}
    else:
        return ham_dict 


def exact_gs(ham_dict):
    H = explicit_hamiltonian(ham_dict)
    #    print(H)
    try:
        w, v = eigh(H)
    except:
        w, v = eig(H)
    multiplicity = list(w).count(w[0])
    return (multiplicity, w[0], v[:, :multiplicity])

def qiskit_dict(ham_dict):
    label_coeff_list = []
    for label, value in ham_dict.items():
        label_coeff_list.append({'label':label,
                                 'coeff':
                                 {'real': value.real, 'imag':value.imag}})
    return {'paulis': label_coeff_list}
