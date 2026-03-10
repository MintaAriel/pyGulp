from pygulp.molecule import fix_mol_gradient
from ase.ga.data import DataConnection
import matplotlib.pyplot as plt
from ase.visualize import view
from huggingface_hub import login

# da = DataConnection('/home/vito/uspex_matlab/theo_pyxtal/test_1/theophilline.db')
da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/database/theophylline_8.db')
connection_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/connections'
atom = da.get_atoms(3)
work_dir  = '/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/Sym'

# atom.cell[1] *= 0.7
# atom.cell[2] *= 0.9
# ASU.cell[0][0] -= 5
# factor = 0.80
# asu.cell[2] *= 0.6
# asu.cell[1] *= factor
# asu.cell[2] *= factor
# # asu.cell[1] *= 0.7
# atom.cell[2][2] -= 3

optimizer = fix_mol_gradient.GradientDescentGULP(atom, work_dir=work_dir, connections=connection_dir)
# optimizer = fix_mol_gradient.GradientDescentUMA(atom, work_dir=work_dir, connections=connection_dir,
#                                                 login_key='insert yours',
#                                                 device='cuda')
# optimizer.eta_t = 20e-3
# optimizer.eta_r = 5e-4

new  = optimizer.run(steps =10,  traj=True)

view(atom)
view(optimizer.best_structure)

steps = [i for i in range(len(optimizer.energies))]
plt.plot(optimizer.energies)
plt.scatter(steps, optimizer.energies)
plt.ylabel('Energy, eV')
plt.xlim(0,)
plt.xlabel('cycle')
plt.show()
