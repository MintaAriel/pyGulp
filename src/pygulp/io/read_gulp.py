import io
import pandas as pd
import numpy as np
import os
import re


def read_results(GULP_output):
    internal_dev = ''
    cell_dev = ''
    energy_lis = []
    volume = None
    with open(GULP_output, 'r') as f:
        text = f.readlines()
        num_lines = len(text)
        for line in range(num_lines):

            if text[line] == '  Final internal derivatives :\n':
                for v in range(line+6, num_lines):
                    if 'Maximum abs' in text[v]:
                        break
                    internal_dev += text[v]

            elif text[line] == '  Final cell parameters and derivatives :\n':
                for v in range(line, num_lines):
                    cell_dev += text[v]
                    if 'Primitive cell' in text[v]:
                        break
            elif 'Total lattice energy' in text[line]:
                if '=' in text[line]:
                    total = text[line].split('=')
                    energy = total[-1].split()[0]
                else:
                    total = text[line+2].split('=')
                    energy = total[-1].split()[0]


                try:
                    energy_lis.append(float(energy))
                except:
                    print('No energy present')
            elif 'cell volume' in text[line]:
                if '=' in text[line]:
                    total = text[line].split('=')
                    volume = total[-1].split()[0]
                else:
                    volume = 0
    # print(internal_dev)

    fixed_inter_dev = re.sub(r'(?<=\d)-(?=\d)', ' -', internal_dev)
    df_gradient = pd.read_csv(io.StringIO(fixed_inter_dev),
                     sep='\s+',
                     skipfooter=1,
                     header=None,
                     engine='python')
    # print(df_gradient)

    df_strain = pd.read_csv(io.StringIO(cell_dev),
                              sep='\s+',
                            skiprows=3,
                              skipfooter=3,
                              header=None,
                            engine='python')

    gradient_array = df_gradient.to_numpy()
    strains_array = df_strain.to_numpy()
    eps = strains_array[:,4]
    eps_tensor = np.array([
        [eps[0], eps[5], eps[4]],
        [eps[5], eps[1], eps[3]],
        [eps[4], eps[3], eps[2]]
    ])

    gradient = gradient_array[:,3:6]


    result = {'strain': eps_tensor,
              'gradient': gradient.astype(float) ,
              'energy': energy_lis[0],
              'volume':volume}


    return result

def consecutive_gulp_read(path_exper):
    experiments = os.listdir(path_exper)
    for experiment in experiments:
        exp_path = os.path.join(path_exper, experiment)
        gulps = os.listdir(exp_path)
        for value in gulps:
            if '.got' in value:
                print(value[-5:-4])
                gulp_got = os.path.join(exp_path, value)
                print(gulp_got)
                with open(gulp_got, 'r') as file:
                    file_content = file.readlines()
                    for index, line in enumerate(file_content):
                        if "**** Too many failed attempts to optimise ****" in line:
                            print('There was an error')
                            f_energy = file_content[index+2]
                            f_Gnorm = file_content[index+3]
                        elif '**** Optimisation achieved ****' in line:
                            print('It was a success')
                            f_energy = file_content[index + 2]
                            f_Gnorm = file_content[index + 3]
                        elif '**** Maximum number of function calls has been reached ****' in line:
                            print('max_num_funct')
                            f_energy = file_content[index + 2]
                            f_Gnorm = file_content[index + 3]
                    print(f' for {value}',f_energy, f_Gnorm)

        print(gulps)

