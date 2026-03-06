import os
import pandas as pd
from io import StringIO
from ase import Atoms
from ase.geometry import Cell
import numpy as np
import re
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from .relax import Gulp_relaxation_noadd
import time
import hashlib
import shutil
import threading


class CifWatcher(threading.Thread):
    """Watches a CIF file for changes and copies it to a numbered backup."""
    def __init__(self, filepath, output_dir):
        super().__init__()
        self.filepath = filepath          # full path to out.cif
        self.output_dir = output_dir      # base directory (self.out_dir)
        self.stop_flag = threading.Event()
        self.last_hash = None
        self.counter = 1

    def run(self):
        # Create structures/ inside output_dir
        structures_dir = os.path.join(self.output_dir, 'structures')
        os.makedirs(structures_dir, exist_ok=True)

        while not self.stop_flag.is_set():
            if os.path.exists(self.filepath):
                # Compute MD5 hash of current file
                with open(self.filepath, 'rb') as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()

                # If file changed (or first time we see it)
                if current_hash != self.last_hash:
                    dest = os.path.join(structures_dir, f'out_{self.counter}.cif')
                    shutil.copy2(self.filepath, dest)
                    # print(f"Saved: {dest}")
                    self.last_hash = current_hash
                    self.counter += 1

            time.sleep(0.005)   # small delay to avoid busy‑waiting

    def stop(self):
        self.stop_flag.set()

def clean_dir(work_dir):
    for item in os.listdir(work_dir):
        if item in 'gulp_parameters':
            print(f"Keeping: {item}")
            continue

        item_path = os.path.join(work_dir, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                print(f"Removed file: {item}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed directory: {item}")
        except Exception as e:
            print(f"Failed to remove {item}: {e}")

class GULP_test_opt:


    def __init__(self, mol_data_dir, out_dir, ase_crystal,  n_mol,atoms_order=True):
        '''
        :param mol_data_dir: directory where the connections of the molecule are (IT SHOULD BE DIFFERENT FROM
        out_dir)
        :param out_dir: working directory for the calculation, traj file
        :param ase_crystal: the ase atoms crystal object to optimize
        :param n_mol: number of molecules in the crystal
        :param atoms_order: True if the order of the atoms is AAABBBCCC, False if the order is ABCABCABC,
        where A,B,C are molecules
        '''
        self.ase_crystal = ase_crystal
        self.mol_data_dir = mol_data_dir
        self.atoms_order = atoms_order
        self.connections = self.mol_data_dir + '/connections'
        self.inter_dis_df = self.read_connections(self.connections)
        self.out_dir = out_dir
        self.n_mol = n_mol
        self.n_atom_per_mol = int(len(self.ase_crystal) / self.n_mol)
        if self.atoms_order == False:
            self.ase_crystal = self.reorder_crystal(self.ase_crystal)

        self.gulp_lib = None

    def reorder_crystal(self, crystal):
        tags = np.array([i for i in range(self.n_mol)] * self.n_atom_per_mol)
        crystal.set_tags(tags)
        final_order = np.argsort(tags, kind='stable')
        new_crystal = crystal[final_order]
        return new_crystal


    def parse_input_file(self, filename):
        """
        Reads a file with -keywords and -options sections.
        Returns a tuple (keywords_str, options_str) containing the lines
        of each section as a single string (preserving original formatting).
        """
        keywords = []
        options = []
        current_section = None

        with open(filename, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped == '-keywords':
                    current_section = 'keywords'
                    continue
                elif line_stripped == '-options':
                    current_section = 'options'
                    continue

                if current_section == 'keywords':
                    keywords.append(line)
                elif current_section == 'options':
                    options.append(line)
                # Lines before any marker are ignored

        return ''.join(keywords), ''.join(options)

    def test_relax(self, bonds: bool = False):
        clean_dir(self.out_dir)
        crystal = self.ase_crystal.copy()
        n_atom_per_mol = len(crystal)/self.n_mol
        bonds_df = self.inter_dis_df
        connections = ''

        if bonds:
            for n in range( self.n_mol):
                for bond in range(len(bonds_df)):
                    delta = n * n_atom_per_mol
                    connections += (
                        f"connect {int(bonds_df['atom1'][bond] + delta)} "
                        f"{int(bonds_df['atom2'][bond] + delta)}\n"
                    )

        else:
            print('Not using bond labeling for GULP optimisation')

        gulp_input, options = self.parse_input_file(self.out_dir + '/gulp_parameters')

        options += f"{connections}\noutput movie cif out.cif\ndump every opt_step"

        relax = Gulp_relaxation_noadd(path=self.out_dir,
                                      library=None,
                                      gulp_keywords=gulp_input,
                                      gulp_options=options)

        out_cif_path = os.path.join(self.out_dir + '/CalcFold', 'opt_step')
        watcher = CifWatcher(out_cif_path, self.out_dir)
        watcher.start()

        try:
            relax.use_gulp(crystal)  # This runs GULP and produces out.cif
        finally:
            watcher.stop()  # Signal the thread to stop
            watcher.join()  # Wait for it to finish


    def convert_to_float(self, value):
        if '/' in str(value):  # Handle fractions like "1/3"
            numerator, denominator = map(float, value.split('/'))
            result = numerator / denominator
        else:
            try:
                result = float(value)
            except:
                return np.nan

        # Add 1 if value is negative
        # if result < 0:
        #     result += 1
        # elif result > 1:
        #     result -= 1
        if result > 1:
            result -= int(result)
        elif result < 0:
            result -= int(result)-1

        return result


    def read_connections(self, connections_txt):
        '''
        read connections as from a txt file
        example:

        connect  1 14
        connect  2 4
        '''

        df_connect = pd.read_csv(connections_txt,
                                 sep='\s+',  # Handles spaces/tabs
                                 header=None,            # No header in file
                                 names=['connect', 'atom1', 'atom2'])
        return df_connect

    def create_traj(self):
        traj_writer = Trajectory(self.out_dir + '/gulp_test_param.traj', 'w')
        num_structures = len(os.listdir(self.out_dir+ '/structures'))
        steps_dir = self.out_dir + '/structures'
        opt_bond_lenght = {}
        atom1 = self.inter_dis_df['atom1']
        atom2 = self.inter_dis_df['atom2']

        for step in range(1,num_structures):
            dir = steps_dir + f'/out_{step}.cif'
            with (open(dir, 'r') as f):
                content = f.readlines() # Read the entire file into a single string
                lines_num = len(content)
                for i in range(lines_num):
                    line = content[i]
                    if line == 'cell \n':
                        cell_parameters = content[i+1].split()
                        cell_vector = Cell.fromcellpar(np.array(cell_parameters, dtype=float))
                    elif line == 'vectors \n':
                        vectors = content[i+1:i+4]
                        cell_vector = np.array([list(map(float, line.split())) for line in vectors])

                    elif line == 'fractional \n':
                        cordinates = content[i + 1:i+len(self.ase_crystal)+1]
                        coordinates_text = ''.join(cordinates)



            # Read with explicit type conversion
            df = pd.read_csv(StringIO(coordinates_text),
                             header=None,
                             sep='\\s+',
                             usecols=[0, 2, 3, 4],
                             converters={2: self.convert_to_float, 3: self.convert_to_float, 4: self.convert_to_float})  #To handle fractions inside the text 1/3

            atoms = df[0].tolist()
            symbols = re.sub(r'\d+', '', ''.join(atoms))
            matrix_data = df.to_numpy()
            positions = matrix_data[:, 1:4]



            crystal = Atoms(symbols=symbols,
                               cell=cell_vector,
                               pbc=True)

            # if ''.join(crystal.symbols) != ''.join(self.ase_crystal.symbols):
            #     crystal = self.reorder_crystal(crystal)
            # print(crystal)

            n_atom_per_mol = int(len(self.ase_crystal) / self.n_mol)

            if len(crystal)  == len(self.ase_crystal) :

                crystal.set_scaled_positions(positions)
                if self.gulp_lib == 'lennard':
                    crystal.set_tags([i for i in range(int(self.n_mol)) for _ in range(n_atom_per_mol)])

                #I think i need to correct self.reorder_crystal, so I can use it here
                elif self.gulp_lib == 'reaxff_general.lib':
                    tags = np.array([i for i in range(self.n_mol)] * self.n_atom_per_mol)
                    crystal.set_tags(tags)

                traj_writer.write(crystal)
                print(f'structure {step} saved')




                single_molecule = (crystal.get_tags() == 0)

                filtered_atom = crystal.__getitem__(single_molecule)


                opt_bond_lenght[f'distance{step}'] = [
                    filtered_atom.get_distance(i - 1, j - 1, mic=True)
                    for i, j in zip(atom1, atom2 )
                ]

            else:
                print(f'out_cif file of step {step} cannot be parsed')

        return opt_bond_lenght


    def plot_bond_change(self,  dic_bonds, bonds):
        # print(tabulate(self.inter_dis_df, headers='keys', tablefmt='psql'))
        df_bonds = pd.DataFrame(dic_bonds)
        distances_array = df_bonds.to_numpy()
        columns = distances_array.shape[1]
        steps = [i for i in range(columns)]
        for bond in bonds:
            distance_by_step = distances_array[bond,:]
            plt.plot(steps, distance_by_step, label=f'bond {bond}')
        plt.title('Bond lenght change by cycle')
        plt.xlabel('Cycle')
        plt.ylabel('Angstrem')
        plt.legend()
        plt.xlim(0,)
        plt.grid()
        plt.savefig(self.out_dir + '/Bond_lenght_change.png')
        plt.show()







