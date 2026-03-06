import os
import sys
from ase.io import read
from ase.calculators.gulp import GULP, GULPOptimizer, Conditions
import spglib
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase_ga.data import DataConnection
from ase.db import connect
import re
import json
import numpy as np
from matplotlib.style.core import library

os.environ["ASE_GULP_COMMAND"]= os.path.expanduser("~/Programs/gulp-6.4/Src/gulp  < PREFIX.gin > PREFIX.got")
os.environ["GULP_LIB"]= os.path.expanduser("/home/vito/Programs/gulp-6.4/Libraries")
#We have to create a customize Gulp class singe we need to define a pressure and
#gulp return 'Total lattice enthalphy' instead of 'Total lattice energy' which GULP class cannot parse



class MyGULP(GULP):



    def read_results(self):
        super().read_results()  # try parent parsing first

        got_path = os.path.join(self.directory, self.label + '.got')
        with open(got_path) as fd:
            lines = fd.readlines()
        self.results['energy'] = 0
        energy = 0

        for line in lines:
            m = re.match(r'\s*Total lattice energy\s*=\s*(\S+)\s*eV', line)

            if m:
                self.results['energy'] = float(m.group(1))
                energy = float(m.group(1))

        return energy

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters

        # Build string to hold .gin input file:
        s = p.keywords
        s += '\ntitle\nASE calculation\nend\n\n'

        if all(self.atoms.pbc):
            cell_params = self.atoms.cell.cellpar()
            # Formating is necessary since Gulp max-line-length restriction
            s += 'cell\n{:9.6f} {:9.6f} {:9.6f} ' \
                 '{:8.5f} {:8.5f} {:8.5f}\n'.format(*cell_params)
            s += 'frac\n'
            coords = self.atoms.get_scaled_positions()
        else:
            s += 'cart\n'
            coords = self.atoms.get_positions()

        if self.conditions is not None:
            c = self.conditions
            labels = c.get_atoms_labels()
            self.atom_types = c.get_atom_types()
        else:
            labels = self.atoms.get_chemical_symbols()

        charges = self.atoms.get_initial_charges()
        n_charges = 0

        for xyz, symbol in zip(coords, labels):
            s += ' {:2} core' \
                 ' {:10.7f}  {:10.7f}  {:10.7f}  {:10.5f}\n' .format(symbol, *xyz, charges[n_charges])
            if symbol in p.shel:
                s += ' {:2} shel' \
                     ' {:10.7f}  {:10.7f}  {:10.7f}\n' .format(symbol, *xyz)
            n_charges += 1



        if p.library:
            s += f'\nlibrary {p.library}\n'

        if p.options:
            for t in p.options:
                s += f'{t}\n'

        gin_path = os.path.join(self.directory, self.prefix + '.gin')
        with open(gin_path, 'w') as fd:
            fd.write(s)

class Gulp_relaxation:

    def __init__(self, path, db_dir):
        self.path = path
        self.db_dir = db_dir
        self.db = connect(self.db_dir)
        self.da = DataConnection(self.db_dir)
        with open(os.path.join(self.path,'labels.json'), 'r') as f:
            self.molecule_labels = json.load(f)
        os.environ["GULP_LIB"] = os.path.expanduser("~/Programs/gulp-6.4/Src/gulp  < PREFIX.gin > PREFIX.got")
        try:
            os.mkdir(os.path.join(self.path, 'CalcFold'))
            print("Directory created!")
        except FileExistsError:
            print("Directory already exists, moving on.")

        os.system(f"cp {os.path.join(self.path,'Specific', 'reaxff_general.lib')} {os.path.join(self.path, 'CalcFold')}")

    def get_symetry(self, atom):
        sg = spglib.get_spacegroup((atom.get_cell(), atom.get_scaled_positions(),
                                    atom.get_atomic_numbers()),
                                   symprec=1e-3)
        if sg is None:
            sg_no = 1
        else:
            sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
        return sg_no

    def number_of_optmi(self, ginput_file, goptions, pressure, atom, library):
        n_of_molecules = atom.get_tags()
        n = np.unique(n_of_molecules)

        labels = Conditions(atom)
        label_suffix = self.molecule_labels*len(n)
        labels.atoms_labels = [f'{value}{label_suffix[index]}' for index, value in enumerate(labels.atoms_symbols)]

        if library != None:
            calculator = MyGULP(keywords=f'{open(goptions).read().strip()} \npressure {pressure} GPa',
                          # goutput file parameters from USPEX
                          options=[open(ginput_file).read().strip()],
                          # maybe Optional Parameters
                          library=library,
                           conditions=labels)
        else:
            calculator = MyGULP(keywords=f'{open(goptions).read().strip()} \npressure {pressure} GPa',
                                # goutput file parameters from USPEX
                                options=[open(ginput_file).read().strip()],
                                # maybe Optional Parameters
                                library=library)

        return calculator

    def use_gulp(self, atom):

        try:
            ginput_dir = os.path.join(self.path,'Specific')
        except:
            print('There is no Specific folder with the Potentials')

        energia = None
        status = None

        library = ['reaxff_general.lib','reaxff_general.lib','reaxff_general.lib',None]
        for i in range(1,2):
            atom.calc = None
            file_name = f'ginput_{i}'
            gulp_path = os.path.join(ginput_dir, file_name)

            goptions_name = f'goptions_{i}'
            goptions_path = os.path.join(ginput_dir, goptions_name)

            calc = self.number_of_optmi(gulp_path, goptions_path, 0, atom, library[i-1])
            calc.prefix = f'ginput{i}'
            calc.directory = os.path.join(self.path, 'CalcFold')
            try:
                GULPOptimizer(atom, calc).run()
                energia =  -calc.get_potential_energy()

            except Exception as e:
                print(e)
                print('Structure not suitable for optimization')
                atom.info['key_value_pairs']['raw_score'] = 0
                self.da.add_relaxed_step(atom)
                self.da.kill_candidate(atom.info['confid'])
                status = 'Failed'
                break

        if status == 'Failed':
            atom.info['key_value_pairs']['spacegroup'] = self.get_symetry(atom)
            atom.info['key_value_pairs']['raw_score'] = energia
            self.da.add_relaxed_step(atom)
            #energia = calc.read_results()

            #atom.calc = None
            #print(f'gulp_{i} is done')

        return atom

    def relax_generation(self):

        while self.da.get_number_of_unrelaxed_candidates() > 0:
            unrelaxed = self.da.get_an_unrelaxed_candidate()
            print('Relaxing starting candidate {}'.format(unrelaxed.info['confid']))
            self.use_gulp(unrelaxed)
            print(self.da.get_number_of_unrelaxed_candidates())

class Gulp_relaxation_noadd:

    def __init__(self, path,  library, gulp_keywords, gulp_options ):
        self.path = path
        self.library = library
        self.gulp_keywords = gulp_keywords
        self.gulp_options = gulp_options
        os.environ["ASE_GULP_COMMAND"]= os.path.expanduser("~/Programs/gulp-6.4/Src/gulp  < PREFIX.gin > PREFIX.got")
        try:
            with open(os.path.join(self.path,'labels.json'), 'r') as f:
                self.molecule_labels = json.load(f)
        except:
            self.molecule_labels = None

        try:
            os.makedirs(os.path.join(self.path, 'CalcFold'))
            print("Directory created!")
        except FileExistsError:
            print("Directory already exists, moving on.")

        if library != None:
            os.system(f"cp {os.path.join('/home/vito/Programs/gulp-6.4/Libraries', f'{self.library}')} {os.path.join(self.path, 'CalcFold')}")

    def get_symetry(self, atom):
        sg = spglib.get_spacegroup((atom.get_cell(), atom.get_scaled_positions(),
                                    atom.get_atomic_numbers()),
                                   symprec=1e-3)
        if sg is None:
            sg_no = 1
        else:
            sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
        return sg_no


    def number_of_optmi(self, keyword, options, atom, library):
        n_of_molecules = atom.get_tags()
        n = np.unique(n_of_molecules)

        labels = Conditions(atom)
        #label_suffix = self.molecule_labels * len(n)

        # labels.atoms_labels = [f'{value}{label_suffix[index]}' for index, value in enumerate(labels.atoms_symbols)]

        if library != None:
            calculator = MyGULP(keywords=keyword,
                                # goutput file parameters from USPEX
                                options=[options],
                                # maybe Optional Parameters
                                library=library,
                                conditions=labels)
        else:
            calculator = MyGULP(keywords=keyword,
                                # goutput file parameters from USPEX
                                options=[options],
                                # maybe Optional Parameters
                                library=library)

        return calculator

    def use_gulp(self, atom):

        try:
            ginput_dir = os.path.join(self.path, 'Specific')
        except:
            print('There is no Specific folder with the Potentials')

        energia = None
        status = None

        library = [self.library, self.library, 'dreiding.lib', None]
        for i in range(1, 2):
            atom.calc = None
            file_name = f'ginput_{i}'
            gulp_path = os.path.join(ginput_dir, file_name)

            goptions_name = f'goptions_{i}'
            goptions_path = os.path.join(ginput_dir, goptions_name)
            #print(self.gulp_options)
            calc = self.number_of_optmi(self.gulp_keywords , self.gulp_options , atom, library[i - 1])
            calc.prefix = f'ginput{i}'

            calc.directory = os.path.join(self.path, 'CalcFold')
            try:
                GULPOptimizer(atom, calc).run()
                # energia = -calc.get_potential_energy()


            except Exception as e:
                print(e)
                print('Structure not suitable for optimization')
                # atom.info['key_value_pairs']['raw_score'] = 0
                status = 'Failed'
                break

        if status == 'Failed':
            print('Error when optimzing using gulp')

        return atom