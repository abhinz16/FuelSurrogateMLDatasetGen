# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:32:26 2024
Last Updated: 08/21/2024

@author: Abhinav Abraham
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def reintialize_mix_ids(num_c, fuel_name, max_iter, DCN=False, MW=False, family_res=False, save_path='', user_name='N/A'):
    """
    Function for re-initializing the mixture ids for consistency...

    Parameters
    ----------
    num_c : int
        The number of component in the mixtures to be considered.
    fuel_name : str
        Name of the target fuel.
    max_iter : int
        Total number of iterations for each n-component mixture.
    DCN : boolean, optional
        True or False. The default is False.
    MW : boolean, optional
        True or False. The default is False.
    family_res : boolean, optional
        True or False. The default is False.
    save_path : str, optional
        Filepath to loaction where generated mixtures are saved. The default is ''.
    user_name : str, optional
        Provide the user name. The default is 'N/A'.

    Returns
    -------
    None.

    """

    surrogate_filepath = Path(save_path).joinpath(f'surrogate_mixtures_{fuel_name}_{max_iter}_iterations_DCN_{DCN}_family_restriction_{family_res}_Number_components_{num_c}_MW_{MW}.csv')
    fragment_filepath = Path(save_path).joinpath(f'surrogate_fragmentation_{fuel_name}_{max_iter}_iterations_DCN_{DCN}_family_restriction_{family_res}_Number_components_{num_c}_MW_{MW}.csv')

    # Skipping inconsistent columns...
    surrogate_file = pd.read_csv(surrogate_filepath, header=None, on_bad_lines='skip')
    no_cols = len(surrogate_file.columns)
    # Reading file again without the inconsistent column...
    surrogate_file = pd.read_csv(surrogate_filepath, header=0, usecols=range(no_cols))

    # Skipping inconsistent columns...
    fragment_file = pd.read_csv(fragment_filepath, header=None, on_bad_lines='skip')
    no_cols = len(fragment_file.columns)
    # Reading file again without the inconsistent column...
    fragment_file = pd.read_csv(fragment_filepath, header=0, usecols=range(no_cols))

    surrogate_file.User_id = user_name
    fragment_file.User_id = user_name

    mixture_ids = ['E'+str(num_c)+str(10000000+i)[1:] for i in range(1,len(fragment_file)+1)]
    fragment_file.mix_id = mixture_ids
    k=0
    for j in range(0, len(surrogate_file)-1, num_c):
        for m in range(j,j+num_c):
            # print(m)
            # print(mixture_ids[k])
            surrogate_file.mix_id[m] = mixture_ids[k]
        k+=1

    surrogate_file.to_csv(surrogate_filepath, index=None)
    fragment_file.to_csv(fragment_filepath, index=None)

def remove_duplicates(num_c, fuel_name, max_iter, DCN=False, MW=False, family_res=False, save_path=''):
    """
    Function to remove duplicates mixtures within an n-component mixture set...

    Parameters
    ----------
    num_c : int
        The number of component in the mixtures to be considered.
    fuel_name : str
        Name of the target fuel.
    max_iter : int
        Total number of iterations for each n-component mixture.
    DCN : boolean, optional
        True or False. The default is False.
    MW : boolean, optional
        True or False. The default is False.
    family_res : boolean, optional
        True or False. The default is False.
    save_path : str, optional
        Filepath to loaction where generated mixtures are saved. The default is ''.

    Returns
    -------
    None.

    """

    def truncate_float(num):
        return float(f"{num:.2f}")

    surrogate_filepath = Path(save_path).joinpath(f'surrogate_mixtures_{fuel_name}_{max_iter}_iterations_DCN_{DCN}_family_restriction_{family_res}_Number_components_{num_c}_MW_{MW}.csv')
    fragment_filepath = Path(save_path).joinpath(f'surrogate_fragmentation_{fuel_name}_{max_iter}_iterations_DCN_{DCN}_family_restriction_{family_res}_Number_components_{num_c}_MW_{MW}.csv')

    appending_surrogate = surrogate_filepath.exists()
    appending_fragment = fragment_filepath.exists()

    if appending_surrogate == True and appending_fragment == True:
        with open(surrogate_filepath,'r') as f1:
            mixture_data = f1.readlines()

        mix_to_drop = []
        for i in range(1, len(mixture_data), num_c):
            if str(mixture_data[i].split(',')[0]) not in mix_to_drop:
                comps_in_mix = [str(mixture_data[c].split(',')[2]) for num, c in enumerate(range(i,i+num_c))]
                weight_frac_in_mix = [str(truncate_float(float(mixture_data[c].split(',')[4]))) for num, c in enumerate(range(i,i+num_c))]
                current_row = comps_in_mix+weight_frac_in_mix
                for j in range(i+num_c, len(mixture_data), num_c):
                    if j!=i and str(mixture_data[j].split(',')[0]) not in mix_to_drop:
                        comps_in_mix = [str(mixture_data[c].split(',')[2]) for num, c in enumerate(range(j,j+num_c))]
                        weight_frac_in_mix = [str(truncate_float(float(mixture_data[c].split(',')[4]))) for num, c in enumerate(range(j,j+num_c))]
                        new_row = comps_in_mix+weight_frac_in_mix
                        if sorted(current_row) == sorted(new_row):
                            for t in range(j, j+num_c, 1):
                                mix_to_drop.append(str(mixture_data[t].split(',')[0]))
                        else: pass
                    else: pass
            else: pass

    surrogate_file = pd.read_csv(surrogate_filepath, header=0, index_col=0)
    fragment_file = pd.read_csv(fragment_filepath, header=0, index_col=0)
    surrogate_file.drop(index=mix_to_drop, inplace=True)
    fragment_file.drop(index=mix_to_drop, inplace=True)

    surrogate_file.to_csv(surrogate_filepath)
    fragment_file.to_csv(fragment_filepath)