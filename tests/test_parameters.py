from ase.io import read
from pGFNFF_opti.relax import Gulp_relaxation_noadd
import numpy as np
from pGFNFF_opti.read_gulp import read_results
from pGFNFF_opti.gnff_fine_tun import crystal_descriptor
from pathlib import Path

script_dir = Path(__file__).parent
# project_root = script_dir.parent
project_root = '/home/vito/PythonProjects/ASEProject/EA/test/POLYMORFS/'


# experimental = read(project_root / 'data/experimental.cif')
experimental = read('/home/vito/Downloads/862238.cif')
# experimental = read('/home/vito/PythonProjects/ASEProject/EA/crystal_ea/utils/0.cif')
def get_parameters( gfnff_param):
    og_crystal = experimental.copy()
    scale_str = ' '.join([str(v)for v in gfnff_param])
    # gulp_input = (f"gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
    #               f"gfnff_scale {scale_str}\n"
    #               f"maths mrrr"
    #               f"pressure 0 GPa"
    #               )
    gulp_input = (f"opti    gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
                  f"gfnff_scale {scale_str}\n"
                  f"maths mrrr\n"
                  f"pressure 0 GPa"
                  )
    options = (
        "output movie cif out1.cif\n"
        "maxcycle 200\n"
        "gtol 0.00001"
    )

    cal_dir = project_root + 'tests/single_param_cal'

    relax = Gulp_relaxation_noadd(path=cal_dir,
                                  library=None,
                                  gulp_keywords=gulp_input,
                                  gulp_options=options)
    atom = relax.use_gulp(og_crystal)
    calc = read_results(f"{cal_dir}/CalcFold/ginput1.got")



    # print(new_param)
    return atom, calc


analyzer = crystal_descriptor(experimental)
initial_vol = experimental.get_volume()

new_crys, crys_param= get_parameters(np.array([ 0.80, 1.343, 0.727, 1.0, 2.859]))
# new_crys, crys_param= get_parameters(np.array([ 0.5915, 1.7055, 0.5463, 0.7126, 0.9907 ]))
# new_crys, crys_param= get_parameters(np.array([ 0.7332, 1.7455, 0.8856, 0.7002, 0.9912 ]))

dV = initial_vol - float(crys_param['volume'])
struc_sim = analyzer.compare(new_crys, 'rmsd')
print('energy', crys_param['energy'])
print(f'rmsd score', struc_sim)
print('volume_dif:', dV)

volume_weight = 0.2
positions_weight = 0.8
# if dV < initial_vol:
#     score = (1 - abs(dV) / initial_vol) * volume_weight + positions_weight * struc_sim
#     print('The score', score, '\n')
# else:
#     score = positions_weight * struc_sim


