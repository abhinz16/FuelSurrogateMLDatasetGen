# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:07:35 2024
Last updated: 04/04/2024

@author: Abhinav Abraham

"""

#%%
import Fragmentation.fg_fragment
from tqdm import tqdm
import glob
import vispy; import logging; vispy.use('pyqt5'); vispy_logger = logging.getLogger('vispy'); vispy_logger.setLevel(logging.CRITICAL)

#%%
config_file_path = r"Z:\PhD\Surrogate development\Surrogate_mixtures\Surrogate_generation_1\Fragmentation\Config.ini"
fragmented_data_fg_path = r"Z:\PhD\Surrogate development\Surrogate_mixtures\Surrogate_generation_1\Fragmentation\Results"

mixture_location = r"Z:\PhD\Surrogate development\Surrogate_mixtures\Surrogate_generation_1\Results\Filtered_3"

files = glob.glob(mixture_location+'/'+'*.csv')
files_mixtures = [f for f in files if f.split('\\')[-1].startswith('surrogate_mixtures')]

for num_comp in tqdm([2,3,4,5,6,7,8,9]):
    PC_comp_save_location_path = [f for f in files_mixtures if int(f.split('\\')[-1].split('_')[-3]) == num_comp][0]

    fragmentation_save_name = f'FG_fragmentation_{num_comp}.csv'

    # Get FG fragmentation of samples...
    print(f'\n\nDetermining FG composition in {num_comp}-component mixtures...\n')
    fg_fragment_class = Fragmentation.fg_fragment.execute(config_file_path, PC_comp_save_location_path, fragmented_data_fg_path, fragmentation_save_name)
    fg_fragment_class.perform(save=True)

