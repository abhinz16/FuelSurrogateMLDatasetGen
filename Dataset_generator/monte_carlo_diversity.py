# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:22:25 2024
Last Updated: 08/21/2024

@author: Abhinav Abraham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import entropy
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Entropy based analysis...
def calculate_entropy(column_data, s, n_bins=50):
    """
    Function to calculate entropy of a dataframe...

    Parameters
    ----------
    column_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    s : int
        The number of mixtures to be considered as a subset.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    entropy_value : numpy float
        The entropy for the dataframe.

    """
    counts, bin_edges  = np.histogram(column_data, bins=n_bins)
    probabilities = counts/len(column_data) # Calculates the probability of a data point falling into each bin by dividing the count of each bin by the total number of data points. This gives you the relative frequency of data points in each bin...
    entropy_value = entropy(probabilities)

    return entropy_value

# monte carlo simulations for finding best subset...
def process_subset_size(s, data, max_iterations=10, num_rows=10, n_bins=50):
    """
    Function to perform Monte-Carlo analysis for each n-component category...

    Parameters
    ----------
    s : int
        The number of mixtures to be considered as a subset.
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of the n-component category under consideration..
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.
    num_rows : int, optional
        The shape of dataframe. The default is 10.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    s: numpy array
        The number of mixtures to be considered as a subset.
    entrops: numpy array
        The mean entropy of each of the subset.

    """

    num_rows = data.shape[0]
    selected_lists = []
    entrops = []

    for _ in range(max_iterations):
        try:
            indices = np.random.choice(num_rows, size=s, replace=False)
            sorted_indices = list(np.sort(indices))
            if sorted_indices not in selected_lists:
                subset = data.iloc[indices]
                selected_lists.append(sorted_indices)

                entropy_values = {}
                for col in subset.iloc[:,3:].columns:
                    entropy_values[col] = calculate_entropy(subset[col], s, n_bins)

                min_entropy = np.min(list(entropy_values.values()))
                entrops.append(min_entropy)

            else:
                _-=1

        except ValueError:
            _-=1
            pass
        except KeyboardInterrupt():
            sys.exit()
        except:
            pass

    return s, np.mean(entrops)

def select_sub_dataset(data, max_iterations=10, n_bins=50):
    """
    Function to perform multiprocessing for monte-carlo analysis...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all n-component category.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    no_of_mixtures : numpy array
        The number of mixtures for each subset..
    entropy_list : numpy array
        A numpy array of the mean entropy for each subset.

    """

    num_rows = data.shape[0]
    subset_size = np.arange(10, 1000, 1)  # Randomly select subset size...

    results = []

    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(process_subset_size, s, data, max_iterations, num_rows, n_bins) for s in subset_size]
        for future in tqdm(as_completed(futures), total=len(subset_size)):
            result = future.result()
            results.append(result)

    no_of_mixtures, entropy_list = zip(*results)

    return no_of_mixtures, entropy_list

def main(file_data, filepath_to_fuel_properties, max_iterations=100, n_bins=50):
    """
    Function to initiate the monte-carlo correlation analysis...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all n-component category.
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 100.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    df_data : dataframe
        A dataframe of number of mixtures and its corresponding entropies.

    """

    filepath_to_fuel_properties = Path(filepath_to_fuel_properties)
    file_data_modified = file_data.drop_duplicates(subset=file_data.columns[3:], keep='first')
    # Check for columns with all zero values and remove them...
    zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
    file_data_modified = file_data_modified.drop(columns=zero_columns)

    # Adjust the dataset...
    no_of_mixtures, entropy_list = select_sub_dataset(file_data_modified, max_iterations=max_iterations, n_bins=n_bins)

    # Correlation...
    fig, axes = plt.subplots()
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    axes.scatter(no_of_mixtures, entropy_list, s=200, c='r')
    try:
        axes.scatter(np.array(no_of_mixtures)[np.where(np.array(entropy_list)==max(entropy_list))[0][0]], max(entropy_list), s=300, c='g', marker='X')
    except:
        raise ValueError('Increase number of iterations or mixtures generated.')
    axes.set_xlabel('Number of mixtures', fontsize=30, fontweight="bold")
    axes.set_ylabel('Entropy', fontsize=30, fontweight="bold")
    axes.set_xticklabels(axes.get_xticks(), ha='center', fontsize=30, fontweight="bold")
    axes.set_yticklabels(axes.get_yticks(), va='center', fontsize=30, fontweight="bold")
    axes.tick_params(left=False, bottom=False)
    axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes.tick_params(axis='both', labelsize=30)
    axes.spines["bottom"].set_linewidth(3)
    axes.spines["top"].set_linewidth(3)
    axes.spines["left"].set_linewidth(3)
    axes.spines["right"].set_linewidth(3)

    plt.savefig(filepath_to_fuel_properties.parent.joinpath("Figures/Mean_minimum_entropy_vs_number_of_mixtures.tif"), dpi=600)
    plt.close()

    df_data = pd.DataFrame({'Number of mixtures': np.array(no_of_mixtures), 'Mean minimum entropy': np.array(entropy_list)})
    df_data.to_csv(filepath_to_fuel_properties.parent.joinpath("Number_of_mixtures_vs_mean_minimum_entropy.csv"), index=0)

    return df_data