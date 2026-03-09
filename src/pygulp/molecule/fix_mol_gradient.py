import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

from ..relaxation.relax import Gulp_relaxation_noadd
from ..io.read_gulp import read_results
import matplotlib.pyplot as plt
from pymatgen.symmetry.groups import SpaceGroup
import pandas as pd
from ase import Atoms
from ase.ga.data import DataConnection
from ase.io.trajectory import Trajectory
from ase.io import write
import os
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter



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
        self.mol_per_ASU = None
        self.mol_per_cell = None


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
        new_symbols = list(self.ASU.symbols) *len(ops)
        new_crystal = Atoms(cell=ASU_new.cell, scaled_positions=new_positions, symbols=new_symbols)

        return new_crystal


    def transform(self, mol_displacement, mol_rotational_matrix, cell_vector):
        '''
        This function transform the Asymmetric unit cell give the displacement in cartesian coordinates, rotational matrix,
        and the new cell vector
        :param
        mol_displacement = [ x, y, z] expressed in Angstrom
        mol_rotation = the rotation of the molecule in cartesian plane, must be 3x3
        cell_transform = the new cell matrix parameters, must be 3x3
        :return: mutated ASU
        '''
        atoms_per_mol = int(len(self.ASU)/ self.mol_per_ASU)
        new_cell = cell_vector

        positions = self.ASU.get_positions().reshape(self.mol_per_ASU,atoms_per_mol,3)
        new_coord_positions = positions + mol_displacement[:, None, :]

        new_R0 = new_coord_positions.mean(axis=1)

        final_positions = []
        for n in range(self.mol_per_ASU):
            R_0 = new_R0[n]
            # rotation matrix
            Q = mol_rotational_matrix[n]

            # move and rotate all atoms
            pos_new = (new_coord_positions[n] - R_0) @ Q.T + R_0
            final_positions.append(pos_new)

        final_positions = np.concatenate(final_positions, axis=0)

        atom_rot = Atoms(cell=new_cell, positions=final_positions, symbols=self.ASU.symbols, pbc=[True, True, True])

        return atom_rot

    def energy_ASU(self, ASU_to_calc, cal_dir, bonds_dir):
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

        mol_in_asym = self.mol_per_ASU
        connections = ''
        # bonds = pd.read_json('bonds.json')
        bonds = read_connections(bonds_dir)
        for n in range(int(mol_in_asym)):
            for bond in range(len(bonds)):
                delta = n * int((len(self.ASU)/self.mol_per_ASU))
                connections += 'connect  ' + str(bonds['atom1'][bond] + delta) + ' ' + str(
                    bonds['atom2'][bond] + delta) + '\n'

        gulp_input = f'gradient conp\npressure {pressure} GPa'

        options = f'spacegroup\n{self.sym_group}\n\n{connections}\noutput movie cif out1.cif\ndump every  optimized.structure paso.cif'

        relax = Gulp_relaxation_noadd(cal_dir, 'reaxff_general.lib', gulp_input, options)
        new = relax.use_gulp(ASU_to_calc)
        # print(new)
        calc = read_results(cal_dir + '/CalcFold/ginput1.got')

        return calc




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


def define_ASU(crystal):
    pmg_structure = AseAtomsAdaptor.get_structure(crystal)
    # Assuming you have a pymatgen Structure object
    analyzer = SpacegroupAnalyzer(pmg_structure)
    similar_atoms = analyzer.get_symmetry_dataset().equivalent_atoms


    unique_items = len(np.unique(similar_atoms))


    print( similar_atoms,unique_items)
    tags = crystal.get_tags()
    tags_unique = tags[np.unique(similar_atoms)]
    asu_tags = np.unique(tags_unique)
    print(asu_tags)

    tags_idx = np.unique(similar_atoms)

    print('tags idx', asu_tags)


    # get_atributes(crystal, asu_tags)

    spacegroup = analyzer.get_symmetry_dataset().number
    print('spacegroup: ', spacegroup)


    n_molecules = len(np.unique(tags))
    n_atoms =  len(crystal)


    mol_in_asym = n_molecules/(n_atoms/unique_items)
    print(mol_in_asym)

    mask = np.isin(tags, asu_tags)
    asu = crystal.__getitem__(mask)
    print(asu)

    # view(asu)

    return   asu, spacegroup, int(mol_in_asym)


def get_cif(atom):

    pmg_structure = AseAtomsAdaptor.get_structure(atom)
    analyzer = SpacegroupAnalyzer(pmg_structure, symprec=1e-3)
    refined_structure = analyzer.get_refined_structure()
    return refined_structure



class gradient_descent():
    def __init__(self, structure, work_dir, connections):
        self.structure = structure
        self.work_dir = work_dir
        self.asu, self.spacegroup, self.n_mol_asu = define_ASU(self.structure)
        self.struc_change = transformASU(self.asu, self.spacegroup)
        self.struc_change.mol_per_ASU = int(self.n_mol_asu)
        self.energies = []
        self.best_structure = None
        self.connections = connections

    def run(self, steps, potential, traj=False):
        #def gradient_descent(ASU, full_cell, potential, mol_per_ASU):
        tags =self.structure.get_tags()
        unique_tags = np.unique(tags)
        n_mol_cell = len(unique_tags)
        atoms_per_mol = int(len(tags)/n_mol_cell)
        self.struc_change.mol_per_cell = n_mol_cell

        displac_grad = np.zeros((self.n_mol_asu, 3))
        rotations = np.repeat(np.eye(3)[None, :, :], int(self.n_mol_asu), axis=0)
        cell_change_0 = self.asu.cell

        if traj:
            trajectory = Trajectory(os.path.join(self.work_dir, f'{potential}_sim.trajectory'), 'w')

        cell_step = 0
        increment = 0.05
        for i in range(steps):
            new_ASU = self.struc_change.transform(mol_displacement=displac_grad,
                                             mol_rotational_matrix=rotations,
                                             cell_vector=cell_change_0)

            if traj:
                #reconstruction of the full cell
                full_new_ASU = self.struc_change.get_full_sym_cell(new_ASU)
                full_new_ASU.set_pbc([True, True, True])
                full_new_ASU.set_tags([i for i in range(n_mol_cell) for _ in range(atoms_per_mol)])
                trajectory.write(full_new_ASU)

            positions = new_ASU.get_positions().reshape(self.n_mol_asu, atoms_per_mol, 3)
            R0 = positions.mean(axis=1)
            R0_expanded = R0[:, None, :]  # shape (n-mol, 1, 3)


            if potential == 'gulp':
                # energy_ASU returns a dict with the strain matrix, gradient of energy per atom, and energy
                try:
                    calculator = self.struc_change.energy_ASU(ASU_to_calc= new_ASU, cal_dir=self.work_dir, bonds_dir= self.connections)
                except Exception as e:
                    print('The symmetry was broken')
                    print(e)
                    break
                # R0 - geometric center
                T = np.linalg.inv(new_ASU.cell).T

                # all forces is and array of len(asu) rows x 3 columns
                # all_forces = -calculator['gradient']
                grad_frac = calculator['gradient']  # from GULP table
                A = new_ASU.cell.array  # 3x3
                all_forces_cart = -(grad_frac @ np.linalg.inv(A))

                forces = all_forces_cart.reshape(self.n_mol_asu, atoms_per_mol, 3)
                gradient_R0 = -np.sum(forces, axis=1)
                # print("Total ASU force:", np.sum(all_forces_cart, axis=0))

                r_rel = positions - R0_expanded  # shape (n-mol, 30, 3)

                torque = np.sum(np.cross(r_rel, forces), axis=1)


                del calculator["gradient"]
                calculator['gradient_r0'] = gradient_R0
                calculator['torque'] = torque
                strain = calculator["strain"]
                eta_t = 80e-4
                eta_r = 5e-6
                eta_c = 500e-8
                print(f'step:{i}    Energy:',calculator['energy'])


                if i == 0:
                    best_energy = calculator['energy']
                    self.best_structure = full_new_ASU

                elif calculator['energy'] < min(self.energies):
                    best_energy = calculator['energy']
                    self.best_structure = full_new_ASU
                    write(os.path.join(self.work_dir, 'best_structure.cif'),self. best_structure)
                    print('This is the lowest so far', best_energy)

                self.energies.append(calculator['energy'])

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

            # Parameters for gradient descent

            # positions
            displac_grad += -eta_t * gradient_R0


            # rotation
            for k in range(self.n_mol_asu):
                rotations[k] = exp_so3(eta_r * torque[k]) @ rotations[k]

            #cell
            # H_new = (np.eye(3) + eta_c * strain ) @ cell_change_0
            # cell_change_0 = H_new
            cell_step +=1
            if cell_step == 3:
                cell_step = 0
                # H_new = (np.eye(3) + eta_c * strain) @ cell_change_0
                # cell_change_0 = H_new
                # print(strain)
                # # struc_change.ASU.cell[0] = b *(1- increment)
                # print(b *(1- increment))
                # increment += 0.05
            else:
                pass

        if traj:
            trajectory.close()





