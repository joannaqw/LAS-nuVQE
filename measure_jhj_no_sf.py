import numpy as np                                                                          
import qiskit
import qiskit_nature
from qiskit.primitives import Estimator,BackendEstimator
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
import math
from qiskit.primitives import Estimator,BackendEstimator
from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit import BasicAer 
from nuVQE import Jastrow_operator,JHJ_operator,JJ_operator
from qiskit.algorithms.optimizers import L_BFGS_B

def qiskit_operator_energy(shots,op,psi,params):
    estimator = BackendEstimator(BasicAer.get_backend('qasm_simulator'))
    job = estimator.run([psi],[op],[params],shots=shots,seed = 0)   
    job_result = job.result()
    return job_result
 
molecule = MoleculeInfo(
    symbols=["H","H"],coords=([0.0, 0.0, 0.0], [0.74, 0.0, 0.0]),multiplicity=1,charge=0)
driver = PySCFDriver.from_molecule(molecule, basis = '6-31g')
problem = driver.run()
num_particles = problem.num_particles
num_spatial_orbitals = problem.num_spatial_orbitals
mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()[0])
hamiltonian = qubit_op
n_qubits = num_spatial_orbitals*2
from qiskit import qpy 
with open('H2_631G_UCC.qpy','rb') as fd: 
    ansatz = qpy.load(fd)[0]
num_Jastrow_params=n_qubits+(n_qubits*(n_qubits-1))//2
jparams =np.load('jastrwo_params.npy').tolist()
vqeparams = np.load('vqe_params.npy').tolist()
params = vqeparams + jparams
num_vqe_params = len(vqeparams)
 
JHJOp=JHJ_operator(params,n_qubits,num_vqe_params,hamiltonian)
 
for i in [10,100,1000,10000,100000,1000000]:
    print(qiskit_operator_energy(i,JHJOp, ansatz, params[:num_vqe_params]))
 

