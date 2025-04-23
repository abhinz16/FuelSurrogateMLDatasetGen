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
from scipy.stats import entropy
import seaborn as sns
import sys
import dcor
from pathlib import Path
import helper_functions
from tqdm import trange
import math
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
    # data_corr = normalized_df.corr(method='spearman')
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

# Entropy based analysis...
def calculate_entropy(column_data, n_bins=50):
    """
    Function to calculate entropy of a dataframe...

    Parameters
    ----------
    column_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    entropy_value : numpy float
        The entropy for the dataframe.

    """

    counts, bin_edges  = np.histogram(column_data, bins=50) # Tells you how many data points fall into each bin...
    probabilities = counts/len(column_data)
    entropy_value = entropy(probabilities)

    return entropy_value

# monte carlo simulations for finding best subset...
def select_sub_dataset(subset_size, data, max_iterations=10, n_bins=50):
    """
    Function to perform multiprocessing for monte-carlo analysis...

    Parameters
    ----------
    subset_size: int
        The number of mixtures to be considered as a subset.
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

    best_subset = None
    best_corr_score = float('inf')
    best_entropy_score = 0.0
    selected_lists = []
    corrs = []
    entrops = []

    for _ in trange(max_iterations):
        try:
            num_rows = np.arange(0, data.shape[0])
            indices = np.random.choice(num_rows, size=subset_size, replace=False)
            sorted_indices = list(np.sort(indices))
            if sorted_indices not in selected_lists:
                subset = data.iloc[indices]
                selected_lists.append(sorted_indices)

                max_correlation = evaluate_maximum(subset.iloc[:,3:])
                corrs.append(max_correlation)

                entropy_values = {}
                for col in subset.iloc[:,3:].columns:
                    entropy_values[col] = calculate_entropy(subset[col], n_bins)
                min_entropy = np.min(list(entropy_values.values()))
                entrops.append(min_entropy)

                if min_entropy > best_entropy_score:
                    best_subset = subset.copy()
                    best_corr_score = max_correlation
                    best_entropy_score = min_entropy
                else: pass

            else:
                _-=1

        except ValueError:
            _-=1
            pass
        except KeyboardInterrupt():
            sys.exit()
        except Exception as e:
            print(e)

    return best_subset, best_corr_score, best_entropy_score, subset_size, np.mean(corrs), np.mean(entrops)

def main(file_data, filepath_to_fuel_properties, fg_prop_path, fuel_name, files_mixtures, optimal_subset_size=50, max_iterations=100):
    """
    Function to initiate the monte-carlo correlation analysis...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all n-component category.
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    fg_prop_path : str
        Filepath to file containing details on functional group properties.
    fuel_name : str
        Name of target fuel.
    files_mixtures: list
        A list of filepaths to surrogate mixtures...
    optimal_subset_size : int, optional
        The optimal subset set size/ minimum number of mixtures for optimal dataset. The default is 50.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 100.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    optimal_dataset : dataframe
        The optimal dataset consisting of surroaget mixture functional group fragmentations.

    """

    filepath_to_fuel_properties = Path(filepath_to_fuel_properties)
    fg_prop_path = Path(fg_prop_path)
    file_data_modified = file_data.drop_duplicates(subset=file_data.columns[3:], keep='first')
    # Check for columns with all zero values and remove them...
    zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
    file_data_modified = file_data_modified.drop(columns=zero_columns)

    # Adjust the dataset...
    optimal_dataset, best_corr_score, best_entropy_score, no_of_mixtures, mean_correlation_value, mean_entropy_value = select_sub_dataset(optimal_subset_size, file_data_modified, max_iterations=max_iterations)

    print(f"\nOptimal number of mixtures: {optimal_subset_size}")
    print(f"Lowest correlation: {best_corr_score}")
    print(f"Highest diversity: {best_entropy_score}\n")
    for l, num_comp in enumerate(file_data.no_components.unique()):
        print('\n'+str(num_comp)+' component mixtures: ', len(optimal_dataset[(optimal_dataset['no_components'] == num_comp)]))

    y = [list(optimal_dataset.loc[:, i].to_numpy()) for i in optimal_dataset.columns[3:]]

    fuel_data = pd.read_csv(filepath_to_fuel_properties, header=0, index_col=0)
    fg_ids = pd.read_csv(fg_prop_path, header=0, index_col=None)
    for old_col in fuel_data.columns:
        if str(old_col).startswith('GUF'):
            fg_id = str(fg_ids.loc[fg_ids.fg_id[fg_ids.fg_id == str(old_col)].index.tolist()[0], 'UNIFAC_fg'])
            fuel_data.rename(columns={old_col: fg_id}, inplace=True)
        else: pass

    fgs = [helper_functions.convert_to_subscript(fg) for fg in optimal_dataset.columns[3:]]

    # Plotting...
    n_cols = math.ceil(math.sqrt(len(optimal_dataset.columns[3:])))
    n_rows = math.ceil(len(optimal_dataset.columns[3:])/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,figsize=(18.2,7.3))
    plt.get_current_fig_manager().window.showMaximized()
    ax = axes.ravel()[:len(optimal_dataset.columns[3:])]

    for h,ax in enumerate(ax):
        # Loop through each subplot and plot the corresponding data...
        if sum(y[h]) != 0:
            sns.distplot(y[h], bins=10, color='green', ax=ax, fit_kws={"color":"white"}, kde=True)
            ax.vlines(x=float(fuel_data.loc[fuel_name, str(optimal_dataset.columns[3:][h])].strip("'[]'").split(', ')[0]),
                      ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
            ax.vlines(x=float(fuel_data.loc[fuel_name, str(optimal_dataset.columns[3:][h])].strip("'[]'").split(', ')[1]),
                      ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
            ax.set_title(str(fgs[h]), fontsize=20, fontweight="bold")
            ax.set_ylabel('Density', fontsize=20, fontweight="bold")
            ax.set_xlabel('Composition', fontsize=20, fontweight="bold")
            ax.tick_params(axis='both', labelsize=20)
            ax.set_yticklabels(ax.get_yticks(), weight='bold')
            ax.set_xticklabels(ax.get_xticks(), weight='bold')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.spines["bottom"].set_linewidth(3)
            ax.spines["top"].set_linewidth(3)
            ax.spines["left"].set_linewidth(3)
            ax.spines["right"].set_linewidth(3)
        else:
            ax.vlines(x=float(fuel_data.loc[fuel_name, str(optimal_dataset.columns[3:][h])].strip("'[]'").split(', ')[0]),
                      ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
            ax.vlines(x=float(fuel_data.loc[fuel_name, str(optimal_dataset.columns[3:][h])].strip("'[]'").split(', ')[1]),
                      ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
            ax.set_title(str(fgs[h]), fontsize=20, fontweight="bold")
            ax.set_ylabel('Density', fontsize=20, fontweight="bold")
            ax.set_xlabel('Composition', fontsize=20, fontweight="bold")
            ax.tick_params(axis='both', labelsize=20)
            ax.set_yticklabels(ax.get_yticks(), weight='bold')
            ax.set_xticklabels(ax.get_xticks(), weight='bold')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.spines["bottom"].set_linewidth(3)
            ax.spines["top"].set_linewidth(3)
            ax.spines["left"].set_linewidth(3)
            ax.spines["right"].set_linewidth(3)

    # Remove empty subplots...
    fig = helper_functions.remove_empty_axes(fig)

    plt.tight_layout()
    plt.savefig(filepath_to_fuel_properties.parent.joinpath("Figures", "Optimal_dataset_fg_distribution_new.tif"), dpi=600)
    plt.savefig(filepath_to_fuel_properties.parent.joinpath("Output", "Optimal_dataset_fg_distribution_new.tif"), dpi=600)
    plt.close()

    # Saving optimal dataset...
    optimal_dataset.sort_index(inplace=True)
    optimal_dataset.to_csv(filepath_to_fuel_properties.parent.joinpath("Output", "Optimal_dataset_FG_fragmentations.csv"), index=None)

    # Generating an excel file containing mixture compositions of each optimal surrogate mixture...
    # Read and combine all CSV files...
    df_list = [pd.read_csv(Path(file)) for file in files_mixtures]  # Read each CSV into a DataFrame...
    combined_df = pd.concat(df_list, ignore_index=True)  # Concatenate them vertically...
    # Save the combined DataFrame into a new CSV file...
    combined_df.to_csv(filepath_to_fuel_properties.parent.joinpath("Output", "combined_wt_fractions_of_mixtures.csv"), index=False)

    optimal_mixtures = combined_df[combined_df.mix_id.isin(optimal_dataset['#'])]
    optimal_mixtures.to_csv(filepath_to_fuel_properties.parent.joinpath("Output", "Optimal_dataset_mixture_compositions.csv"), index=None)

    return optimal_dataset, optimal_mixtures