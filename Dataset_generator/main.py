# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:18:21 2024

PROCESS FLOW:
1) Check libraries and the folders available.
2) Perform fragmentation of palette components to get FG counts.
3) Generate surrogate mixtures.
4) Post processing of the generated mixtures to ensure correct mixture ids.
5) Remove temporarily created files.
6) Generating a csv file combining the FGs of mixtures for statistical analysis later.
7) Create FG distribution plots.
8) Correlation  and entropy analysis --> how does the metrics change with the number of components and number of mixtures.
9) Monte-Carlo analysis (irrespective of number of components) --> how does correlation and entropy vary with number of mixtures.
10) Find the minimum number of mixtures for an optimal dataset. Threshold provided by the user as an input at this step.
11) Generate optimal dataset for the minimum number of mixtures identified using Monte-Carlo analysis.
12) Finally, create the necessary files and figures.
    
DO NOT NAME FUELS WITH '_' IN THEM.
RUN THE CODE AND ASSOCIATED FILE FROM A LOCAL DRIVE. TRY TO AVOID NETWORK DRIVES AND SUCH.

REPLACE THE PALETTE OF COMPONENTS YOU WOULD LIKE TO USE IN 'Components_with_SMILES.csv' FILE.
PROVIDE THE UNIFAC FUNCTIONAL GROUP COMPOSITIONS OF FUEL IN 'Fuel_props.csv' FILE.

@author: Abhinav Abraham

"""

# Import libraries...
import vispy; import logging; vispy.use('pyqt5'); vispy_logger = logging.getLogger('vispy'); vispy_logger.setLevel(logging.CRITICAL)
import sys
import numpy as np
import pandas as pd
import surrogate_generator
from tqdm import tqdm
import os
import fg_fragmenter
from platform import python_version
import matplotlib
import seaborn
import scipy
import postprocessing
import glob
import warnings
warnings.filterwarnings('ignore')

# Provide the location to store the data...
folderpath = ""
sys.path.append(folderpath)

if __name__=="__main__":
    if python_version() > '3.10.13' or str(sys.version_info[0]) < str(3):
        print('Installed python version: ', python_version())
        raise Exception('This code has only been tested on python version 3.10.13 or lower and is python 3.x. Please install python version 3.10.13 or lower and is python 3.x...')
    if pd.__version__ > '1.5.3':
        print('Installed pandas version: ', pd.__version__)
        raise Exception('This code has only been tested on pandas version 1.5.3 or lower. Please install pandas version 1.5.3 or lower...')
    if matplotlib.__version__ > '3.8.3':
        print('Installed matplotlib version: ', matplotlib.__version__)
        raise Exception('This code has only been tested on matplotlib version 3.8.3 or lower. Please install matplotlib version 3.8.3 or lower...')
    if seaborn.__version__ > '0.13.2':
        print('Installed seaborn version: ', seaborn.__version__)
        raise Exception('This code has only been tested on seaborn version 0.13.2 or lower. Please install seaborn version 0.13.2 or lower...')
    if scipy.__version__ > '1.12.0':
        print('Installed seaborn version: ', scipy.__version__)
        raise Exception('This code has only been tested on scipy version 1.12.0 or lower. Please install scipy version 1.12.0 or lower...')
    else:
        # Provide user name here...
        user_name = 'akayam2'

        # Filepath to where the files with information on the fuel properties and component properties are stored...
        filepath_to_fuel_properties = os.path.join(folderpath, "Fuel_props.csv")
        filepath_to_PC_info = os.path.join(folderpath, "Components_with_SMILES.csv")
        fg_prop_path = os.path.join(folderpath, "fg_properties.csv")
        if not os.path.exists(os.path.join(folderpath, "Figures")):
            os.mkdir(os.path.join(folderpath, "Figures"))
        if not os.path.exists(os.path.join(folderpath, "Results")):
            os.mkdir(os.path.join(folderpath, "Results"))
        if not os.path.exists(os.path.join(folderpath, "Output")):
            os.mkdir(os.path.join(folderpath, "Output"))
        save_location = os.path.join(folderpath, "Results")

        # Name of the target fuel...
        fuel_details = pd.read_csv(filepath_to_fuel_properties, header=0, index_col=0)
        fuel_name = str(fuel_details.index[0]) # The target fuel should be in 'filepath_to_fuel_properties'...
        # Number of components to be in each surrogate mixture. Provide number of components in a list...
        num_components = [1,2,3,4,5,6]
        # Provide the threshold below which the component is considered as trace component...
        threshold = 0.01

        # Provide the desired constraints...
        DCN_constr = False
        MW_constr = False
        family_restriction = False

        # Provide the maximum number of iterations...
        max_iter = 10000
        # Provide the number of mixtures that the prgram should try to find for each n-component category mixtures...
        max_mixtures = 50

        # Statistical analysis paramters...
        # Number of constant bins for entropy calculations...
        n_bins = 50
        # Correlation and entropy monte-carlo analysis for each n-component category...
        max_iterations_each = 500
        # Correlation and entropy monte-carlo analysis irrespective of n-component category...
        max_iterations_full = 500

        # To find optimal dataset...
        max_iterations_optimal = 500

        print('\n\nFragmentation of pure components...\n')
        FG_class = fg_fragmenter.execute(filepath_to_PC_info, fg_prop_path, filepath_to_fuel_properties)
        PC_data = FG_class.perform()

        print('\n\nGenerating mixtures...\n')
        surrogate_generator.main(num_components, filepath_to_fuel_properties, PC_data, fuel_name, threshold, save_location, DCN_constr, MW_constr, family_restriction, max_iter, max_mixtures, user_name)
        surrogate_generator.rearrange_files(fuel_name, num_components, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location)

        # Post processing of the generated mixtures to remove duplicates...
        files = glob.glob(os.path.join(save_location, '*.csv'))
        files_mixtures = [f for f in files if f.split('/')[-1].startswith('surrogate_mixtures') or f.split('\\')[-1].startswith('surrogate_mixtures')]
        files_fragmentation = [f for f in files if f.split('/')[-1].startswith('surrogate_fragmentation') or f.split('\\')[-1].startswith('surrogate_fragmentation')]

        print('\n\nPost-processing mixtures - Part 1...\n')
        for num_comp in tqdm(num_components):
            postprocessing.reintialize_mix_ids(num_comp, fuel_name, max_iter, DCN_constr, MW_constr, family_restriction, save_location, user_name)

        print('\n\nPost-processing mixtures - Part 2...\n')
        for num_comp in tqdm(num_components):
            postprocessing.remove_duplicates(num_comp, fuel_name, max_iter, DCN_constr, MW_constr, family_restriction, save_location)
            postprocessing.reintialize_mix_ids(num_comp, fuel_name, max_iter, DCN_constr, MW_constr, family_restriction, save_location, user_name)

        print("\n\nRemoving temporary files...\n")
        surrogate_generator.remove_temp_files(fuel_name, num_components, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location)

        # Import libraries...
        import helper_functions
        import plot_fg_distribution
        import correlation
        import pairsplot
        import diversity
        import monte_carlo_correlation
        import monte_carlo_diversity
        import optimal_number_of_mixtures
        import optimal_dataset_generator

        # Combine functional group fragmentations for all surrogate mixtures...
        print('\n\nGenerating combined FGs of mixtures file for statistical analysis...\n')
        filenames = np.array([os.path.join(save_location, k) for k in os.listdir(save_location) if k.startswith(('surrogate_fragmentation')) and k.endswith('.csv')])
        combined_fgs_mixtures = helper_functions.combine_fg_data(filenames, fuel_name, fg_prop_path, PC_data, user_name)
        combined_fgs_mixtures.drop(columns=[combined_fgs_mixtures.columns[-1]], inplace=True)

        # Make plots of the above created dataframe...
        print('\n\nCreating and saving FG distribution plots...\n')
        plot_fg_distribution.main(num_components, fuel_name, filepath_to_fuel_properties, fg_prop_path, combined_fgs_mixtures)

        """ Statistical Analysis..."""
        # Correlation analysis...
        print('\n\nPerforming correlation analysis...\n')
        x_list, y_list = correlation.main(filepath_to_fuel_properties, combined_fgs_mixtures, max_iterations=max_iterations_each)
        helper_functions.plotting_correlation(filepath_to_fuel_properties, combined_fgs_mixtures, x_list, y_list)
        pairsplot.main(combined_fgs_mixtures, filepath_to_fuel_properties)

        # Diversity analysis...
        print('\n\nPerforming diversity analysis...\n')
        diversity.summary(combined_fgs_mixtures, filepath_to_fuel_properties)
        diversity.entropy_calculation(combined_fgs_mixtures, filepath_to_fuel_properties, fg_prop_path, n_bins)
        diversity.PCA_analysis(combined_fgs_mixtures, fg_prop_path)
        diversity.main(filepath_to_fuel_properties, combined_fgs_mixtures, fg_prop_path, max_iterations=max_iterations_each, n_bins=n_bins)

        """ Monte-Carlo analysis..."""
        print('\n\nMonte-carlo analysis for correlation...\n')
        correlation_data = monte_carlo_correlation.main(combined_fgs_mixtures, filepath_to_fuel_properties, max_iterations_full)
        print('\n\nMonte-carlo analysis for diversity...\n')
        entropy_data = monte_carlo_diversity.main(combined_fgs_mixtures, filepath_to_fuel_properties, max_iterations_full, n_bins)

        """ Optimal number of mixtures..."""
        print('\n\nFinding optimal number of mixtures...\n')
        intersect = optimal_number_of_mixtures.get_diff(filepath_to_fuel_properties, correlation_data, entropy_data)
        suitable_threshold = False
        while not suitable_threshold:
            try:
                """Enter the threshold based on the difference plot. Keep trying new ones if no intersection found.
                     Refer the plot obtained for the precious line to find sutiable threshold..."""
                threshold = float(input("\nEnter threshold for finding minimum number of mixtures for optimal dataset: "))
                minimum_no_of_mixtures = optimal_number_of_mixtures.main(filepath_to_fuel_properties, correlation_data, entropy_data, threshold, intersect)
                suitable_threshold = True
            except:
                pass

        """ Generating optimal dataset... """
        print('\n\nGenerating optimal dataset...\n')
        files = glob.glob(os.path.join(save_location, '*.csv'))
        files_mixtures = [f for f in files if f.split('/')[-1].startswith('surrogate_mixtures') or f.split('\\')[-1].startswith('surrogate_mixtures')]
        files_fragmentation = [f for f in files if f.split('/')[-1].startswith('surrogate_fragmentation') or f.split('\\')[-1].startswith('surrogate_fragmentation')]
        optimal_dataset, optimal_mixtures = optimal_dataset_generator.main(combined_fgs_mixtures, filepath_to_fuel_properties, fg_prop_path, fuel_name, files_mixtures, optimal_subset_size=minimum_no_of_mixtures, max_iterations=max_iterations_optimal)
