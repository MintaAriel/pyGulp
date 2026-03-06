import numpy as np
from relax import Gulp_relaxation_noadd
from read_gulp import read_results
import matplotlib.pyplot as plt
from pymatgen.symmetry.groups import SpaceGroup
from ase.geometry import cell_to_cellpar, cellpar_to_cell
import pandas as pd
from ase import Atoms
from ase.ga.data import DataConnection
from ase.io.trajectory import Trajectory
from scipy.optimize import minimize

from ase.visualize import view
from ase.io import write
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupOperations
from sympy.physics.vector import gradient
import math
from ase.optimize.optimize import Optimizer





def transform_cell(orig_cell, delta_params):
    """
    Transform an ASE cell by adding deltas to lengths and angles while preserving orientation.

    :param orig_cell: original ASE cell, 3x3 array
    :param delta_params: array-like [Δa, Δb, Δc, Δα, Δβ, Δγ] (angles in degrees)
    :return: new ASE 3x3 cell matrix
    """
    # Original cell parameters
    orig_params = cell_to_cellpar(orig_cell)  # [a, b, c, α, β, γ]

    # New cell parameters
    new_params = orig_params + np.array(delta_params)

    # Compute scaling factors for each vector
    scales = new_params[:3] / orig_params[:3]  # a, b, c

    # Scale original vectors
    new_cell = orig_cell.copy() * scales[:, np.newaxis]

    # Adjust angles using rotation if needed
    # ASE's cellpar_to_cell can generate canonical vectors, but to preserve orientation we apply small correction
    orig_angles = orig_params[3:]
    new_angles = new_params[3:]

    # Only adjust if angles are changed
    if not np.allclose(orig_angles, new_angles):
        # Compute a canonical cell from new_params
        canonical_cell = cellpar_to_cell(new_params)

        # Compute rotation that aligns original a vector with canonical a vector
        a_orig = new_cell[0] / np.linalg.norm(new_cell[0])
        a_can = canonical_cell[0] / np.linalg.norm(canonical_cell[0])

        # Axis-angle rotation
        axis = np.cross(a_orig, a_can)
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            axis = axis / norm
            angle = np.arccos(np.clip(np.dot(a_orig, a_can), -1.0, 1.0))

            # Rodrigues rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            new_cell = (R @ new_cell.T).T

    return new_cell

def euler_to_rot(self, alpha, beta, gamma):
    # Rotation around Z
    Rz1 = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha),  np.cos(alpha), 0],
                    [0,              0,             1]])
    # Rotation around Y
    Ry  = np.array([[ np.cos(beta), 0, np.sin(beta)],
                    [0,             1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])
    # Rotation around Z again
    Rz2 = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma),  np.cos(gamma), 0],
                    [0,              0,             1]])
    return Rz1 @ Ry @ Rz2


bonds = pd.read_json('bonds.json')



class transformASU:
    """
    Rigid-body transformer for an asymmetric unit (single molecule).
    """

    def __init__(self, ASU: Atoms, sym_group: int):
        self.ASU = ASU
        self.sym_group = sym_group

        # fixed reference geometry
        self.ref_positions = ASU.get_positions().copy()
        self.ref_R0 = self.ref_positions.mean(axis=0)

    def transform(self, mol_transform, mol_rotational_matrix, cell_vector):
        """
        BFGS-safe rigid-body transform.

        Parameters
        ----------
        mol_transform : (3,)
            Cartesian translation (Å)
        mol_rotational_matrix : (3,3)
            Infinitesimal rotation matrix (I + skew(w))
        cell_vector : (3,3)
            Cell matrix

        Returns
        -------
        Atoms
        """

        pos0 = self.ref_positions
        R0 = self.ref_R0

        # rotate about fixed center, then translate
        pos_rot = (pos0 - R0) @ mol_rotational_matrix.T + R0
        pos_new = pos_rot + mol_transform

        return Atoms(
            cell=cell_vector,
            positions=pos_new,
            symbols=self.ASU.symbols,
            pbc=True
        )

    def get_full_sym_cell(self, ASU_new):
        """
        Build full crystal from asymmetric unit using space-group symmetry.
        """

        sg = SpaceGroup.from_int_number(self.sym_group)

        frac = ASU_new.get_scaled_positions()
        all_pos = []
        all_sym = []

        for op in sg.symmetry_ops:
            W = op.rotation_matrix
            t = op.translation_vector

            p = frac @ W.T + t
            p %= 1.0

            all_pos.append(p)
            all_sym.extend(ASU_new.symbols)

        return Atoms(
            cell=ASU_new.cell,
            scaled_positions=np.vstack(all_pos),
            symbols=all_sym,
            pbc=True
        )

def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def exp_so3_small(w):
    """First-order SO(3) approximation: valid for |w| << 1"""
    return np.eye(3) + skew(w)

def energy_ASU(ASU_to_calc, space_group, cal_dir):
    pressure = 0
    maxcycl = 1

    mol_in_asym = 1
    connections = ''

    for n in range(mol_in_asym):
        for bond in range(len(bonds)):
            delta = n * len(ASU_to_calc)
            connections += (
                f"connect {bonds['aid1'][bond] + delta} "
                f"{bonds['aid2'][bond] + delta}\n"
            )

    gulp_input = f"gradient conp\npressure {pressure} GPa"

    options = (
        f"spacegroup\n{space_group}\n\n"
        f"{connections}\n"
        "output movie cif out1.cif\n"
        "dump every optimized.structure paso.cif"
    )

    relax = Gulp_relaxation_noadd(
        cal_dir,
        "reaxff_general.lib",
        gulp_input,
        options
    )

    relax.use_gulp(ASU_to_calc)
    calc = read_results(f"{cal_dir}/CalcFold/ginput1.got")

    return calc


da = DataConnection('/home/vito/PythonProjects/ASEProject/CARLO/Carbamazepine/carbamazepine.db')
#  da = DataConnection('/home/vito/Downloads/Mg4Al8O16_401.db')
#da = DataConnection('/home/vito/PythonProjects/ASEProject/CARLO/theophylline/theophylline_8.db')
atom = da.get_atoms(6)
tags = atom.get_tags()
symbol = atom.symbols
mask = np.isin(tags, 0)
ASU = atom.__getitem__(mask)

# ASU.cell[0][0] -= 5
# ASU.cell[1][1] -= 5
# ASU.cell[2][2] -= 5
ASU_spacegroup = 32
struc_change = transformASU(ASU, ASU_spacegroup)
print(ASU.cell)
# full = struc_change.get_full_sym_cell(ASU)


# from fairchem.core import pretrained_mlip, FAIRChemCalculator
# from huggingface_hub import login

# predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda", cache_dir='/home/vito/Programs/UMA_models')
# calc = FAIRChemCalculator(predictor, task_name="omat")

cell_change_0 = ASU.cell
energies = []
traj = Trajectory(f'/home/vito/PythonProjects/ASEProject/CARLO/theophylline/CalcFold/bfgs_sim.traj', 'w')


def objective_bfgs(x, potential="gulp"):
    """
    x = [tx, ty, tz, wx, wy, wz]
    """

    dt = x[:3]      # Cartesian translation
    dw = x[3:]      # infinitesimal rotation

    Q = exp_so3_small(dw)

    # rigid transform
    new_ASU = struc_change.transform(
        mol_transform=dt,
        mol_rotational_matrix=Q,
        cell_vector=cell_change_0
    )

    # energy + gradients
    calc = energy_ASU(
        new_ASU,
        ASU_spacegroup,
        "/home/vito/PythonProjects/ASEProject/CARLO/theophylline"
    )

    # fractional → Cartesian
    T = np.linalg.inv(new_ASU.cell).T
    forces = -calc["gradient"] @ T

    pos = new_ASU.get_positions()
    R0 = pos.mean(axis=0)

    # rigid-body gradients
    grad_t = -np.sum(forces, axis=0)
    grad_w = -np.sum(np.cross(pos - R0, forces), axis=0)

    grad = np.hstack([grad_t, grad_w])

    return calc["energy"], grad

scale = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])

def obj_scaled(y):
    x = y * scale
    E, g = objective_bfgs(x)
    return E, g * scale

y0 = np.zeros(6)

res = minimize(
    obj_scaled,
    y0,
    jac=True,
    method="BFGS",options=dict(gtol=1e-4, maxiter=50)
)

x_opt = res.x * scale

print("Optimised [t_cart, w] =", x_opt)
print("Final energy =", res.fun)
