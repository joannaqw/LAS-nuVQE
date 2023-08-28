import numpy as np
import logging
import time
from argparse import ArgumentParser
from typing import Tuple, List
import itertools
# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo
from pyscf.tools import fcidump
# mrh imports
#from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasci import h1e_for_las
from mrh.exploratory.unitary_cc import uccsd_sym1, lasuccsd
# Qiskit imports
import qiskit
import qiskit_nature
from qiskit.providers.aer import StatevectorSimulator, QasmSimulator
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
from qiskit.utils import QuantumInstance
from qiskit.algorithms import NumPyEigensolver, VQE 
from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA, BOBYQA
from qiskit.opflow import PauliTrotterEvolution,SummedOp,PauliOp,MatrixOp,PauliSumOp,StateFn
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
# Qiskit imports
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
qiskit_nature.settings.use_pauli_sum_op = False
#for nu-VQE:
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal

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
          # list_paulis.append({'label': string_paulis,'coeff': {'real': -params[self.num_vqe_params+ind_string], 'imag': 0.0}})
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
    #pseudo_inv_JOp = np.linalg.pinv(JOp)
    #JHJOp = JOp.dot(hamiltonian).dot(pseudo_inv_JOp)
    #JHJOp = JOp*hamiltonian*JOp
    return JHJOp

def JJ_operator(params,n_qubits,num_vqe_params):
    JOp=Jastrow_operator(params,n_qubits,num_vqe_params)
    JJOp=JOp.dot(JOp)
    #JJOp = JOp*JOp
    return JJOp

def qiskit_operator_energy(n_qubits,params,qubitOp,psi):
    #backend=Aer.get_backend(backend)
    estimator = Estimator()
    job = estimator.run([psi], [qubitOp], [params])
    job_result = job.result()

    return job_result
         
####################################################below is self-defined##########################

#Do nuVQE with HF, PySCFdiver
molecule = MoleculeInfo(
    symbols=["H","H"],coords=([0.0, 0.0, 0.0], [0.74, 0.0, 0.0]),multiplicity=1,charge=0)
driver = PySCFDriver.from_molecule(molecule, basis = '6-31g')

        # Get properties
problem = driver.run()

num_particles = problem.num_particles
num_spatial_orbitals = problem.num_spatial_orbitals

mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()[0])

hamiltonian = qubit_op
n_qubits = 8
n_blocks = 2
num_vqe_params =((n_blocks+1)*n_qubits) 
Jastrow_initial=0.1
num_Jastrow_params=n_qubits+(n_qubits*(n_qubits-1))//2
params=(np.random.uniform(0., 2.*np.pi, size=num_vqe_params)).tolist()+(np.random.uniform(-Jastrow_initial,Jastrow_initial,size=num_Jastrow_params)).tolist()
init = HartreeFock(num_spatial_orbitals,num_particles,mapper)
ansatz = TwoLocal(n_qubits, 'ry', 'cx', 'linear', reps=n_blocks, insert_barriers=True, initial_state =init )
#optimizer = L_BFGS_B(maxfun=10000, iprint=101)
#init_pt = np.zeros(num_vqe_params)

def cost_func(params,n_qubits,num_vqe_params,hamiltonian,ansatz):
    JHJOp=JHJ_operator(params,n_qubits,num_vqe_params,hamiltonian)                                                             
    JJOp=JJ_operator(params,n_qubits,num_vqe_params)
    numerator=qiskit_operator_energy(n_qubits,params[:num_vqe_params],JHJOp,ansatz)
    denominator=qiskit_operator_energy(n_qubits,params[:num_vqe_params],JJOp,ansatz)
    energy_qiskit=numerator.values/denominator.values
    return energy_qiskit
# Running the VQE
t0 = time.time()
res = minimize(cost_func,params,args = (n_qubits,num_vqe_params,hamiltonian,ansatz),method='L-BFGS-B',tol=0.00001,options={'disp':True})
t1 = time.time()
print("Time taken for VQE: ",t1-t0)
print("VQE energies: ", values)
#test = cost_func(params,n_qubits,num_vqe_params,hamiltonian,ansatz)
#print(test)

