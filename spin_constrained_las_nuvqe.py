import numpy as np  
import time
# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo
from pyscf.tools import fcidump
# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasci import h1e_for_las
from mrh.my_pyscf.tools import molden
#from pyscf.tools import molden
# Qiskit imports
import qiskit
import qiskit_nature
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
qiskit_nature.settings.use_pauli_sum_op = False
from qiskit.circuit.library import RealAmplitudes
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from initialize_las import initialize_las
from get_hamiltonian import get_hamiltonian
from nuVQE import Jastrow_operator,JHJ_operator,JJ_operator

def qiskit_operator_energy(params,qubitOp,psi):
    estimator = Estimator()
    job = estimator.run([psi], [qubitOp], [params])
    job_result = job.result() 
    return job_result
mapper = JordanWignerMapper()
from qiskit_nature.second_q.properties.s_operators import s_minus_operator, s_plus_operator,s_z_operator,FermionicOp
def total_spin_penalty(num_spatial_orb,ansatz,params,spin_ideal):
    S_p =  s_plus_operator(num_spatial_orb)
    S_m= s_minus_operator(num_spatial_orb)
    Sz = s_z_operator(num_spatial_orb)
    total_S = S_p@S_m + (Sz)**2 - Sz
    num_spin_orbitals = 2*num_spatial_orb
    identity_dict = {}
    for i in range(num_spin_orbitals):
        identity_dict[f"+_{i} -_{i}"] = 1.0 
        identity_dict[f"-_{i} +_{i}"] = 1.0 
    identity_op = FermionicOp(identity_dict,num_spin_orbitals=2*num_spatial_orb)
    total_S -= spin_ideal*identity_op
    total_S_pen = (total_S)**2
    total_S_q = mapper.map(total_S_pen)
    pen = qiskit_operator_energy(params,total_S_q,ansatz).values[0]
    return pen 

xyz = '''C          0.72800       -0.72800        0.00000
C          0.72800        0.72800        0.00000
C         -0.72800       -0.72800        0.00000
C         -0.72800        0.72800        0.00000
H          1.48390       -1.48390        0.00000
H          1.48390        1.48390        0.00000
H         -1.48390       -1.48390        0.00000
H         -1.48390        1.48390        0.00000'''
mol = gto.M(atom = xyz, basis = 'cc-pvdz',output='c4h4_ccpvdz.log',symmetry=False, verbose=lib.logger.DEBUG)

mf = scf.RHF(mol).run()
#mo = np.load('true_gs_rhf.npy')
print("RHF energy: ", mf.e_tot)
las = LASSCF(mf, (2,2),((2,0),(0,2)), spin_sub=(3,3))
frag_atom_list = ((0,1),(2,3))
loc_mo_coeff = las.localize_init_guess(frag_atom_list,mf.mo_coeff)
las.kernel(loc_mo_coeff)
print("LASSCF energy: ", las.e_tot)
mc  = mcscf.CASCI(mf, 4,(2,2))
mc.kernel(las.mo_coeff)
print("CASCI energy: ", mc.e_tot) 
ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
# CASCI h1 & h2 for VQE Hamiltonian
loc_mo_coeff = las.mo_coeff
mc = mcscf.CASCI(mf,4,(2,2))
mc.kernel(loc_mo_coeff)
cas_h1e, e_core = mc.h1e_for_cas()
eri_cas = mc.get_h2eff(loc_mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)
hamiltonian = get_hamiltonian(None, mc.nelecas, mc.ncas, cas_h1e, eri)
n_qubits =int(np.sum(ncas_sub)*2)
n_blocks = 2
num_vqe_params =((n_blocks+1)*n_qubits) #this is for ry+linear
print("vqe params:", num_vqe_params)
Jastrow_initial=0.1
num_Jastrow_params=n_qubits+(n_qubits*(n_qubits-1))//2
params=(np.random.uniform(0., 2.*np.pi, size=num_vqe_params)).tolist()+(np.random.uniform(-Jastrow_initial,Jastrow_initial,size=num_Jastrow_params)).tolist()
new_circuit = initialize_las(las)
ansatz = TwoLocal(n_qubits, 'ry', 'cx', 'linear', reps=n_blocks, insert_barriers=False,initial_state = new_circuit)

mapper = JordanWignerMapper() 
from qiskit_nature.second_q.properties.s_operators import s_minus_operator, s_plus_operator,s_z_operator
def total_spin(num_spatial_orb,ansatz,params):
    S_p =  s_plus_operator(num_spatial_orb)
    S_m= s_minus_operator(num_spatial_orb)
    Sz = s_z_operator(num_spatial_orb)
    total_S = S_p@S_m + (Sz)**2 + Sz
    total_S_q = mapper.map(total_S)
    S_squared = qiskit_operator_energy(params,total_S_q,ansatz)
    return S_squared
 

def cost_func(params,n_qubits,num_vqe_params,hamiltonian,ansatz):
    JHJOp=JHJ_operator(params,n_qubits,num_vqe_params,hamiltonian)
    JJOp=JJ_operator(params,n_qubits,num_vqe_params)
    numerator=qiskit_operator_energy(params[:num_vqe_params],JHJOp,ansatz)
    denominator=qiskit_operator_energy(params[:num_vqe_params],JJOp,ansatz)
    energy_qiskit=numerator.values/denominator.values
    S_pen = total_spin_penalty(int(n_qubits/2),ansatz,params[:num_vqe_params],spin_ideal=0)
    mu = 1
    return energy_qiskit + mu*S_pen

t0 = time.time()
res = minimize(cost_func,params,args = (n_qubits,num_vqe_params,hamiltonian,ansatz),method='L-BFGS-B',options={'disp':True})
t1 = time.time()
print("Time taken for VQE: ",t1-t0)
print('vqe result', res.x)
print('total spin is:', total_spin(4,ansatz,res.x[:num_vqe_params]))      
