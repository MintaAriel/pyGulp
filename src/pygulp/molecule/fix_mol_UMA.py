
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from huggingface_hub import login
import numpy as np
from ase.visualize import view
from ase.io import read
import importlib.resources
from ase.ga.data import DataConnection





predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda", cache_dir='/home/vito/Programs/UMA_models')
calc = FAIRChemCalculator(predictor, task_name="omat")

# #atoms = bulk("Fe")
# #
#atom_unrel = da.get_atoms(3)
#
# view(atoms)
# atoms.calc = calc
da = DataConnection('/home/vito/uspex_matlab/theo_pyxtal/test_1/theophilline.db')
atom_unrel = da.get_atoms(5)


# opt = FIRE(FrechetCellFilter(atom_unrel), trajectory='/home/vito/PythonProjects/ASEProject/CARLO/reaf_fix_mol.traj')
# opt.run(0.005, 100)
#UMA strain may be in Stress tensor (σ), a 2nd rank tensor in GPa. It's force per unit area 1 GPa·Å³ = 0.006241509 eV

atom_unrel.calc = calc
volume = atom_unrel.get_volume()
gpa2ev = 0.006241509
atom_unrel.get_forces()
results = atom_unrel.calc.results
eps = results['stress']*gpa2ev * -volume
results['strain'] = np.array([
        [eps[0], eps[5], eps[4]],
        [eps[5], eps[1], eps[3]],
        [eps[4], eps[3], eps[2]]
    ])

# forces = atom_unrel.get_forces()
# print(f"Forces (shape: {forces.shape}):\n", forces)
print(results)
print(results['energy'])

forces = results['forces']
block1 = np.absolute(forces[0:30, :] )  # Rows 0-29 (Python is 0-indexed, so 1-30 in human terms)
block2 = np.absolute(forces[30:60, :])  # Rows 30-59
block3 = np.absolute(forces[60:90, :])  # Rows 60-89
block4 = np.absolute(forces[90:120, :]) # Rows 90-119

dif2 = block1 - block2
dif3 = block1 - block3
dif4 = block1 - block4


print(f"Block 1 shape: {block1}")  # Should be (30, 3)
print(f"Block 2 shape: {dif2}")  # Should be (30, 3)
print(f"Block 3 shape: {dif3}")  # Should be (30, 3)
print(f"Block 4 shape: {dif4}")  # Should be (30, 3)

print(forces.shape)
