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
hcoeffs = JHJOp.coeffs 
#UDS: strategy 1: measure all Paulis independently, each with Nshot/N shots
for i in [1850,18500,185000,1850000,18500000,185000000]:
    print('shots assigned:', i)
    total_shots = i 
    prob_shots = np.abs(hcoeffs) / np.sum(np.abs(hcoeffs))
    import math
    uds = np.full(shape = len(JHJOp),fill_value=math.floor(total_shots/len(hcoeffs))) 
    print("STRATEGY 1 --- UNIFORM SAMPLING  ")
    total_e = 0.0 
    sig_i = []
    h_i = []
    for op,sii in zip(JHJOp, uds):
        total_e += qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).values[0]
        h_i.append(qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).values[0])
        sig_i.append(qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).metadata[0]['variance'])
    print('UDS energy:', total_e)
    var_u =  (np.sum(np.abs(hcoeffs))/total_shots)*np.sum([x*y for x, y in zip(np.abs(JHJOp.coeffs),sig_i)])
    print('UDS variance:', var_u)

#WDS: strategy 2: measure all Paulis independently, with ci/m shots 
hcoeffs = JHJOp.coeffs   
for i in [1850,18500,185000,1850000,18500000,185000000]:
    print('shots assigned:', i)
    total_shots = i 
    prob_shots = np.abs(hcoeffs) / np.sum(np.abs(hcoeffs))
    wds =np.array([math.floor(x) for x in prob_shots *total_shots])
    print("STRATEGY 2 --- WEIGHTED DETERMINISTIC SAMPLING ")
    mask = wds != 0
    new_ham =JHJOp[mask]
    total_e = 0.0 
    sig_i = []
    h_i = []
    for op,sii in zip(new_ham, wds[mask]):
        total_e += qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).values[0]
        h_i.append(qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).values[0])
        sig_i.append(qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).metadata[0]['variance'])
#    print(qiskit_operator_energy(sii,op,ansatz,params))
    print('WDS energy:', total_e)   
    var_d =  (np.sum(np.abs(hcoeffs))/total_shots)*np.sum([x*y for x, y in zip(np.abs(new_ham.coeffs),sig_i)])
    print('WDS variance:', var_d)
 
#WRS: strategy 3:  measure all Paulis independently, with random sampling 
hcoeffs = JHJOp.coeffs   
for i in [1850,18500,185000,1850000,18500000,185000000]:
    print('shots assigned:', i)
    total_shots = i
    prob_shots = np.abs(hcoeffs) / np.sum(np.abs(hcoeffs))g
    print("STRATEGY 3 --- WEIGHTED RANDOM SAMPLING ")
    from scipy.stats import multinomial
    np.random.seed(0)
    si = multinomial(n=total_shots, p=prob_shots)
    wrs = si.rvs()[0]
    mask = wrs != 0
    new_ham = JHJOp[mask]
    total_e = 0.0
    sig_i = []
    h_i = []
    for op,sii in zip(new_ham, wrs[mask]):
        total_e += qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).values[0]
        h_i.append(qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).values[0])
        sig_i.append(qiskit_operator_energy(sii,op,ansatz,params[:num_vqe_params]).metadata[0]['variance'])
    print('WRS energy:', total_e)
    var_d =  (np.sum(np.abs(hcoeffs))/total_shots)*np.sum([x*y for x, y in zip(np.abs(new_ham.coeffs),sig_i)])
    var_r = var_d + (np.sum(np.abs(hcoeffs))/total_shots)*np.sum([x*(y**2) for x,y in zip(np.abs(new_ham.coeffs),h_i)]) - (qiskit_operator_energy(None,hamiltonian,ansatz,params[:num_vqe_params]).values[0])**2/total_shots
    print('WRS variance:',var_r)
