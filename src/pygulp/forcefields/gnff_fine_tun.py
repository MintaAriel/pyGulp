from ase.io import read
from dscribe.descriptors import SOAP
import optuna
from .relax import Gulp_relaxation_noadd
import numpy as np
from .read_gulp import read_results


class crystal_descriptor():
    def __init__(self, atom):
        self.atom  = atom
        self.soap_des = self.use_SOAP(self.atom)
        self.rmsd_des = self.use_RMSD(self.atom)

    def use_SOAP(self, ase_atom):
        soap = SOAP(
            species=["C", "H", "O", "N"],  # Elements in your system
            r_cut=5.0,  # Cutoff radius (Å)
            n_max=8,  # Radial basis functions
            l_max=6,  # Angular basis functions
            sigma=0.2,  # Gaussian smearing width
            rbf="gto",  # Radial basis function type
            periodic=True,  # CRITICAL for crystals!
            average="inner",  # 'off' for per-atom, 'inner' for global
            sparse=False
        )

        # Compute SOAP descriptors
        # Returns: (n_atoms, n_features) matrix
        s1 = soap.create(ase_atom)
        return s1

    def use_RMSD(self, ase_atom):
        fractional_coord = ase_atom.get_positions(wrap=True)
        return fractional_coord

    def compare(self, atom, type):

        if type == 'soap':
            s1 = self.soap_des
            s2 = self.use_SOAP(atom)
            dot = np.dot(s1, s2)
            k = dot ** 2 / np.sqrt((np.dot(s1, s1) ** 2) * (np.dot(s2, s2) ** 2))

        elif type == 'rmsd':
            s1 = self.rmsd_des
            s2 = self.use_RMSD(atom)
            df = s1 - s2
            rmsd = np.sqrt((df**2).sum(axis=1).mean())
            k = 1 - rmsd*0.1

        return k

def get_parameters( gfnff_param, molcrys, calc_dir):
    og_crystal = molcrys.copy()
    scale_str = ' '.join([str(v)for v in gfnff_param])
    # gulp_input = (f"gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
    #               f"gfnff_scale {scale_str}\n"
    #               f"maths mrrr"
    #               f"pressure 0 GPa"
    #               )
    gulp_input = (f"opti    gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
                  f"gfnff_scale {scale_str}\n"
                  f"maths mrrr"
                  f"pressure 0 GPa"
                  )
    options = (
        "output movie cif out1.cif\n"
        "maxcycle 20"
    )

    relax = Gulp_relaxation_noadd(path=calc_dir,
                                  library=None,
                                  gulp_keywords=gulp_input,
                                  gulp_options=options)
    atom = relax.use_gulp(og_crystal)
    calc = read_results(f"{calc_dir}/CalcFold/ginput1.got")

    return atom, calc




def tune_gfnff(delta_par, db_name, n_trials, fingerprint, project_dir, low_boundry=None, high_boundry=None):
    '''

    :param delta_par: the +- limit of the parameters for the optimisation
    :param db_name: name of the sqlite db where the results are stored
    :param n_trials: max amount of trials
    :param fingerprint: rmsd or soap descriptor
    :param low_boundry:
    :param high_boundry:
    :return:
    '''
    experimental = read(project_dir / 'data/experimental.cif')
    gfnff_scale = np.array([0.546335, 1.823183, 0.368031, 0.100, 2.440330])
    gfnff_my = np.array([0.70, 1.343, 0.727, 1.0, 1.41455])

    analyzer = crystal_descriptor(experimental)
    initial_vol = experimental.get_volume()
    print(initial_vol)

    ini_par = np.array([ 0.80, 1.343, 0.727, 1.0, 1.41455 ])
    # mcGFN-FF gfnff_scale s_coulomb s_rep s_hb s_c6 s_c8

    if low_boundry and high_boundry:
        pass
    else:
        low_boundry = [v*(1-delta_par) for v in ini_par]
        high_boundry = [v*(1+delta_par) for v in ini_par]

    def objective(trial):

        theta = np.array([
            trial.suggest_float("p1", low_boundry[0], high_boundry[0]),
            trial.suggest_float("p2", low_boundry[1], high_boundry[1]),
            trial.suggest_float("p3", low_boundry[2], high_boundry[2]),
            trial.suggest_float("p4", low_boundry[3], high_boundry[3]),
            trial.suggest_float("p5", low_boundry[4], high_boundry[4]),
        ])

        calc_dir = project_dir / 'tests/opt_calcs'

        try:
            new_crys, crys_param = get_parameters(theta, experimental, calc_dir)
            relax_stat = True
        except Exception as e:
            print(e)
            relax_stat = False

        if relax_stat == True:
            dV = initial_vol - float(crys_param['volume'])
            struc_sim = analyzer.compare(new_crys, fingerprint)

            print('energy', crys_param['energy'])
            print(f'{fingerprint} score', struc_sim)
            print('volume_dif:', dV)

            volume_weight = 0.2
            positions_weight = 0.8

            if dV < initial_vol:
                score = (1 - abs(dV) / initial_vol) * volume_weight + positions_weight * struc_sim
                print('The score', score, '\n')
            else:
                score = positions_weight * struc_sim
        else:
            score = 0.5
            print('bad parameters')
        return score

    db_path = project_dir / 'results' / f"{db_name}.db"
    study_name = db_name
    storage_name = f"sqlite:///{db_path}"

    # optuna.delete_study(
    #     study_name=study_name,
    #     storage=storage_name
    # )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True  # Continue if study already exists
    )
    # study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)
    print(study.best_value)



