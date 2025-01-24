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
#jparams =np.load('jastrwo_params.npy').tolist()
jparams = [-0.05000021896285059, 0.03359095621763289, -0.08125899580584031, 0.09847689128939338, 0.030126472706351437, 0.060409675257398876, 0.03288333699296336, -0.049527387223110744, -0.07786537277388145, 0.005427710046347234, -0.04906994671842351, -0.03630143800969421, -0.0882589868050173, 0.031970754269526835, 0.0315051053186009, -0.018754348152682926, 0.06500604366139345, -0.05596059359850636, -0.08952980796636555, -0.011097013879295486, -0.0537255840667942, 0.009376754395377954, -0.03909772936441123, 0.024738850502462648, -0.017575754007724195, -0.07609669489070994, -0.08895075206376157, 0.07535532692775293, 0.040326150988649195, -0.07439766305685509, 0.09857947880507731, -0.08629294724700985, -0.07655021685313296, -0.08178585720578349, -0.03644083582526925, -0.014946520525152551]
vqeparams = np.load('vqe_params.npy').tolist()
params = vqeparams + jparams
num_vqe_params = len(vqeparams)
 
JHJOp=JHJ_operator(params,n_qubits,num_vqe_params,hamiltonian)
 
for i in [10,100,1000,10000,100000,1000000]:
    print(qiskit_operator_energy(i,JHJOp, ansatz, params[:num_vqe_params]))
 

