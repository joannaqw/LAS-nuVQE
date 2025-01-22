import numpy as np 
import qiskit
import qiskit_nature
from qiskit.primitives import Estimator,BackendEstimator
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

def Jastrow_operator(params,n_qubits,num_vqe_params):
    # Calculate the Pauli matrix strings that compose the Jastrow operator J
    string_paulis='I'*n_qubits
    list_paulis=[{'label': string_paulis,'coeff': 1.0}]
    for ind_string in range(n_qubits):
        string_paulis=''
        for ind_position in range(n_qubits):
            if ind_position==ind_string:
                string_paulis+='Z'
            else:
                string_paulis+='I'
        list_paulis.append({'label': string_paulis,'coeff': -params[num_vqe_params+ind_string]})
    counter=0
    for ind_string_1 in range(n_qubits):
        for ind_string_2 in range(ind_string_1+1,n_qubits):
            string_paulis=''
            for ind_position in range(n_qubits):
                if ind_position==ind_string_1 or ind_position==ind_string_2:
                    string_paulis+='Z'
                else:
                    string_paulis+='I'
            list_paulis.append({'label': string_paulis,'coeff': -params[num_vqe_params+n_qubits+counter]})
            counter+=1
    p_list = []
    coeff = []
    for i in range(len(list_paulis)-1):
        p_list.append(list_paulis[i+1]['label'])
        coeff.append(list_paulis[i+1]['coeff'])
    JOp=SparsePauliOp(p_list,coeff)
    return JOp

def JHJ_operator(params,n_qubits,num_vqe_params,hamiltonian):
    JOp=Jastrow_operator(params,n_qubits,num_vqe_params)
    JHJOp=JOp.dot(hamiltonian).dot(JOp.conjugate())
    return JHJOp.simplify()

def JJ_operator(params,n_qubits,num_vqe_params):
    JOp=Jastrow_operator(params,n_qubits,num_vqe_params)
    JJOp=JOp.dot(JOp)
    return JJOp.simplify()

def qiskit_operator_energy(n_qubits,params,qubitOp,psi):
    #backend=Aer.get_backend(backend)
    estimator = Estimator()
    job = estimator.run([psi], [qubitOp], [params])
    job_result = job.result()



