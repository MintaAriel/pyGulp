from pygulp.molecule import fix_mol_gradient
from ase.ga.data import DataConnection
import matplotlib.pyplot as plt
from ase.visualize import view

da = DataConnection('/home/vito/uspex_matlab/theo_pyxtal/test_1/theophilline.db')
connection_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/connections'
atom = da.get_atoms(2)
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

optimizer = fix_mol_gradient.gradient_descent(atom, work_dir=work_dir, connections=connection_dir)
new  = optimizer.run(steps =100, potential='gulp', traj=True)

view(atom)
view(optimizer.best_structure)

steps = [i for i in range(len(optimizer.energies))]
plt.plot(optimizer.energies)
plt.scatter(steps, optimizer.energies)
plt.ylabel('Energy, eV')
plt.xlim(0,)
plt.xlabel('cycle')
plt.show()
