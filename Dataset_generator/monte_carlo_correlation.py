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
import dcor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def distance_corr_dataframe(data, min_periods: int = 1):
    """
    Function to find distance correlation for a dataframe...

    Parameters
    ----------
    data : dataframe
        A normalized pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of an n-component category.
    min_periods : int, optional
        The default is 1.

    Returns
    -------
    data_dist_corr : dataframe
        A dataframe of distance correlations of given dataframe.

    """

    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T
    K = len(cols)
    data_dist_corr = pd.DataFrame(index=data.columns, columns=data.columns)
    mask = np.isfinite(mat)
    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue

            valid = mask[i] & mask[j]
            if valid.sum() < min_periods:
                c = np.nan
            elif not valid.all():
                c = dcor.distance_correlation(ac[valid], bc[valid])
            else:
                c = dcor.distance_correlation(ac, bc)
            data_dist_corr.iloc[i, j] = c
            data_dist_corr.iloc[j, i] = c

    # Replace all NaN by 0 in the correlation dataframe...
    data_dist_corr.fillna(value=0, inplace=True)

    return data_dist_corr

def evaluate_correlation(data):
    """
    Function to normalize the dataframe and find distance correlation...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of an n-component category.

    Returns
    -------
    data_corr : dataframe
        A dataframe of distance correlations of given dataframe.

    """

    # Normalize each column to have zero mean and unit variance to eliminate effect of sample size...
    normalized_df = (data - data.mean())/data.std()
    data_corr = distance_corr_dataframe(normalized_df)
    data_corr.fillna(value=0, inplace=True)
    # Replace diagonal elements with zeros...
    np.fill_diagonal(data_corr.values, 0)

    return data_corr

def evaluate_maximum(data):
    """
    Function to find the maximum correlation in a dataframe of distance correlations between variables...

    Parameters
    ----------
    data : dataframe
        A dataframe of distance correlations of given dataframe.

    Returns
    -------
    data_max : numpy float
        The maximum correlation in a dataframe of distance correlations.

    """

    data_corr = evaluate_correlation(data)
    data_max = data_corr.abs().values.max()
    absolute_max_index = data_corr.abs().values.argmax()

    return data_max

# monte carlo simulations for finding best subset...
def process_subset_size(s, data, max_iterations=10, num_rows=10):
    """
    Function to perform Monte-Carlo analysis for each n-component category...

    Parameters
    ----------
    s : int
        The number of mixtures to be considered as a subset.
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all n-component category.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.
    num_rows : int, optional
        The shape of dataframe. The default is 10.

    Returns
    -------
    s: numpy array
        The number of mixtures to be considered as a subset.
    corrs: numpy array
        A numpy array of mean of correlation of maximum iterations of each subset.

    """

    selected_lists = []
    corrs = []

    for _ in range(max_iterations):
        try:
            indices = np.random.choice(num_rows, size=s, replace=False)
            sorted_indices = list(np.sort(indices))
            if sorted_indices not in selected_lists:
                subset = data.iloc[indices]
                selected_lists.append(sorted_indices)

                evenness = evaluate_maximum(subset.iloc[:,3:])
                corrs.append(evenness)

            else:
                _-=1

        except ValueError:
            _-=1
            pass
        except KeyboardInterrupt():
            sys.exit()

    return s, np.mean(corrs)

def select_sub_dataset(data, max_iterations=10):
    """
    Function to perform multiprocessing for monte-carlo analysis...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all n-component category.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.

    Returns
    -------
    no_of_mixtures: numpy array
        Numpy array of the subsets.
    correlation_list: Numpy array
        A numpy array of correlations for each subset.

    """

    num_rows = data.shape[0]
    subset_size = np.arange(10, 1000, 1)  # Randomly select subset size...

    results = []

    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(process_subset_size, s, data, max_iterations, num_rows) for s in subset_size]
        for future in tqdm(as_completed(futures), total=len(subset_size)):
            result = future.result()
            results.append(result)

    no_of_mixtures, correlation_list = zip(*results)

    return no_of_mixtures, np.array(correlation_list)

def main(file_data, filepath_to_fuel_properties, max_iterations=10):
    """
    Function to initiate the monte-carlo correlation analysis...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all n-component category.
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.

    Returns
    -------
    df_data : dataframe
        A dataframe of number of mixtures and its corresponding correlations.

    """

    filepath_to_fuel_properties = Path(filepath_to_fuel_properties)
    file_data_modified = file_data.drop_duplicates(subset=file_data.columns[3:], keep='first')
    # Check for columns with all zero values and remove them...
    zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
    file_data_modified = file_data_modified.drop(columns=zero_columns)

    # Adjust the dataset...
    no_of_mixtures, correlation_list = select_sub_dataset(file_data_modified, max_iterations=max_iterations)

    fig, axes = plt.subplots()
    axes.scatter(no_of_mixtures, np.array(correlation_list), s=200, c='m')
    axes.set_xlabel('Number of mixtures', fontsize=30, fontweight="bold")
    axes.set_ylabel('Correlation', fontsize=30, fontweight="bold")
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

    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()

    plt.savefig(filepath_to_fuel_properties.parent.joinpath("Figures/Mean_maximum_distance_correlation_vs_number_of_mixtures.tif"), dpi=600)
    plt.close()

    df_data = pd.DataFrame({'Number of mixtures': np.array(no_of_mixtures), 'Mean maximum correlation': np.array(correlation_list)})
    df_data.to_csv(filepath_to_fuel_properties.parent.joinpath("Number_of_mixtures_vs_mean_maximum_correlation.csv"), index=0)

    return df_data