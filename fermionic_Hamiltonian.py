#from qiskit.chemistry.drivers import PyQuanteDriver
#from qiskit.chemistry import FermionicOperator
from qiskit_nature.second_q.drivers import PySCFDriver

class fermionic_Hamiltonian:
    def __init__(self,molecule):
   #     driver=PyQuanteDriver(atoms=molecule['atoms'],charge=molecule['charge'],multiplicity=molecule['multiplicity'],basis=molecule['basis'])
        driver = PySCFDriver(atoms=molecule['atoms'], basis = molecule['basis'],charge=molecule['charge'],spin=molecule['multipicity'])
        self.mol = driver.run()
        self.num_particles = self.mol.num_particles
       # self.num_particles = self.mol.num_alpha + self.mol.num_beta
        self.num_spin_orbitals = self.mol.num_spatial_orbitals * 2
        self.ferOp = self.mol.hamiltonian.second_q_op()
       # self.ferOp = FermionicOperator(h1=self.mol.one_body_integrals, h2=self.mol.two_body_integrals)
