import numpy as np
from ..relaxation.relax import Gulp_relaxation_noadd
from ..io.read_gulp import read_results
import matplotlib.pyplot as plt
from pymatgen.symmetry.groups import SpaceGroup
from ase.geometry import cell_to_cellpar, cellpar_to_cell
import pandas as pd
from ase import Atoms
from ase.ga.data import DataConnection
from ase.io.trajectory import Trajectory






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

class transformASU():
    '''
    Transform_ASU object

    Attributes
    ---------------------
    ASU: ASE Atoms
        This is the Asymmetric unit where we assume that there is 1 molecule that will be transformed
    sym_group: str
        the spacegroup of the crystal  # Example: P6_3/mmc (hexagonal)
    '''
    def __init__(self, ASU:Atoms, sym_group):
        self.ASU = ASU
        self.sym_group = sym_group

    def get_full_sym_cell(self, ASU_new):
        '''
        computes the full cell using symmetric operations of its spacegroup.
        :param ASU_new: ASE atoms cell with the asymmetric cell
        :return: full cell
        '''

        sg = SpaceGroup.from_int_number(self.sym_group)

        ASU_positions = ASU_new.get_scaled_positions()

        all_positions = []

        ops = list(sg.symmetry_ops)
        for i, op in enumerate(ops):
            op = ops[i]
            W = op.rotation_matrix
            t = op.translation_vector

            pos_sym = ASU_positions @ W.T + t
            pos_sym %= 1
            all_positions.append(pos_sym)

        new_positions = np.concatenate(all_positions, axis=0)
        new_symbols = list(ASU.symbols) *4
        new_crystal = Atoms(cell=ASU_new.cell, scaled_positions=new_positions, symbols=new_symbols)

        return new_crystal


    def transform(self, mol_transform, mol_rotational_matrix, cell_vector):
        '''
        This function transform the Asymmetric unit cell give the displacement in cartesian coordinates, rotational matrix,
        and the new cell vector
        :param
        mol_transform = [ x, y, z] expressed in Angstrom
        mol_rotation = the rotation of the molecule in cartesian plane, must be 3x3
        cell_transform = the new cell matrix parameters, must be 3x3
        :return: mutated ASU
        '''

        new_cell = cell_vector
        trans_displacement = mol_transform
        ASU_coord_positions = self.ASU.get_positions() + trans_displacement
        R_0 = ASU_coord_positions.mean(axis=0)
        anchor = R_0

        # new desired center in fractional coordinates
        R = anchor

        # rotation matrix
        Q = mol_rotational_matrix

        # move and rotate all atoms
        pos_new = (ASU_coord_positions - anchor) @ Q.T + R
        atom_rot = Atoms(cell=new_cell, positions=pos_new, symbols=self.ASU.symbols, pbc=[True, True, True])

        return atom_rot

def read_connections( connections_txt):
    '''
    read connections as from a txt file
    example:

    connect  1 14
    connect  2 4
    '''

    df_connect = pd.read_csv(connections_txt,
                             sep='\s+',  # Handles spaces/tabs
                             header=None,  # No header in file
                             names=['connect', 'atom1', 'atom2'])
    return df_connect


def energy_ASU(ASU_to_calc, space_group, cal_dir):
    '''
    This function calls GULP to compute the gradients of the energy and the stress in the cell, the gradients are in
    fractional coordinates, it returns a dictionary with results.
    :param ASU_to_calc: asymmetric cell to calculate
    :param space_group: spacegroup of the full cell
    :param cal_dir: directory to save the gulp input and output files
    :return:
    '''
    pressure = 0
    maxcycl = 1

    mol_in_asym=1
    connections = ''
    bonds = read_connections('/home/vito/PythonProjects/ASEProject/EA/data/carbamazepine/data/connections')
    for n in range(int(mol_in_asym)):
        for bond in range(len(bonds)):
            delta = n * 30
            connections += 'connect  ' + str(bonds['atom1'][bond] + delta) + ' ' + str(
                bonds['atom2'][bond] + delta) + '\n'
    gulp_input = f'gradient conp\npressure {pressure} GPa'

    options = f'spacegroup\n{space_group}\n\n{connections}\noutput movie cif out1.cif\ndump every  optimized.structure paso.cif'


    relax = Gulp_relaxation_noadd(cal_dir, 'reaxff_general.lib', gulp_input, options)
    new = relax.use_gulp(ASU_to_calc)
    #print(new)
    calc = read_results(cal_dir+'/CalcFold/ginput1.got')

    return calc

def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def exp_so3(w):
    '''
    Returns a rotational matrix given an array of the torque in x,y,z in R0
    :param w: skew matrix
    :return: rotational matrix 3x3
    '''
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    W = skew(w/theta)
    return (
        np.eye(3)
        + np.sin(theta) *W
        + (1- np.cos(theta)) * (W @ W)
    )


da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/data/carbamazepine/database/carbamazepine.db')
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


energies = []
def gradient_descent(potential):



    rot_mat_0 = np.eye(3) #no rotation
    cell_change_0 = ASU.cell

    displac_grad = np.array([0, 0, 0] , dtype=float)
    traj = Trajectory(f'/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/sym-old/{potential}_sim.traj', 'w')
    for i in range(40):
        new_ASU = struc_change.transform(mol_transform=displac_grad,
                                         mol_rotational_matrix=rot_mat_0,
                                         cell_vector=cell_change_0)
        full_new_ASU = struc_change.get_full_sym_cell(new_ASU)
        full_new_ASU.set_pbc([True, True, True])
        full_new_ASU.set_tags([i for i in range(4) for _ in range(int(len(tags)/4))])
        traj.write(full_new_ASU)
        # view(new_ASU)
        # energy_ASU returns a dict with the strain matrix, gradient of energy per atom, and energy
        positions = new_ASU.get_positions()
        R0 = positions.mean(axis=0)


        if potential == 'gulp':
            calculator = energy_ASU(new_ASU, ASU_spacegroup, '/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/sym-old')

            # R0 - geometric center
            T = np.linalg.inv(new_ASU.cell).T
            forces = -calculator['gradient']
            gradient_R0 = -np.sum(forces, axis=0)
            print(gradient_R0)
            torque = np.sum(np.cross(positions - R0, forces), axis=0)
            del calculator["gradient"]
            calculator['gradient_r0'] = gradient_R0
            calculator['torque'] = torque
            strain = calculator["strain"]
            eta_t = 20e-4
            eta_r = 5e-6
            eta_c = 100e-6
            print(f'step:{i}    Energy:',calculator['energy'])
            energies.append(calculator['energy'])

        elif potential == 'uma':
            calc = None
            full_new_ASU.calc = calc
            volume = full_new_ASU.get_volume()
            gpa2ev = 0.006241509
            full_new_ASU.get_forces()

            results = full_new_ASU.calc.results
            eps = results['stress'] * gpa2ev * -volume
            strain = np.array([
                [eps[0], eps[5], eps[4]],
                [eps[5], eps[1], eps[3]],
                [eps[4], eps[3], eps[2]]
            ])

            forces = results['forces'][0:30, :]
            gradient_R0 = -np.sum(forces, axis=0)
            torque = np.sum(np.cross(positions - R0, forces), axis=0)
            eta_t = 20e-4
            eta_r = 5e-6
            eta_c = 5e-3
            print(f'step:{i}    Energy:', results['energy'])



        #Parameters for gradient descent
        #positions
        t_new = R0 - eta_t*gradient_R0
        delta_t = -eta_t*gradient_R0
        displac_grad += np.array(delta_t)

        #rotation
        delta_w = - eta_r * - torque
        R_new = exp_so3(delta_w) @ rot_mat_0
        rot_mat_0 = R_new

        #cell
        # H_new = (np.eye(3) + eta_c * strain ) @ cell_change_0
        #cell_change_0 = H_new

    traj.close()


gradient_descent('gulp')
steps = [i for i in range(len(energies))]
plt.plot(energies)
plt.scatter(steps, energies)
plt.xlabel('Energy, eV')
plt.xlim(0,)
plt.ylabel('cycle')
plt.show()
