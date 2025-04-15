# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:09:53 2024
Last Updated: 08/21/2024

@author: Abhinav Abraham
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import helper_functions
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import matplotlib.font_manager as font_manager
import warnings
warnings.filterwarnings("ignore")

# Get a summary of the dataset...
def summary(file_data, filepath_to_fuel_properties):
    """
    Function to generate a summary of dataframe of each n-component mixtures...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.

    Returns
    -------
    None.

    """

    filepath_to_fuel_properties = Path(filepath_to_fuel_properties)
    for num_comp in file_data.no_components.unique():
        file_data_modified = file_data[(file_data['no_components'] == num_comp)]
        file_data_modified = file_data_modified.drop_duplicates(subset=file_data_modified.columns[3:], keep='first')
        # Check for columns with all zero values and remove them...
        zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
        file_data_modified = file_data_modified.drop(columns=zero_columns)

        # Calculate summary statistics...
        summary = file_data_modified.iloc[:, 3:-1].describe().T  # Transpose for easier plotting...

        # Plot summary statistics...
        n_cols = math.ceil(math.sqrt(len(summary.columns)))
        n_rows = math.ceil(len(summary.columns)/n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
        fig.suptitle(f'Summary Statistics for {num_comp}-component mixtures', fontsize=25, fontweight="bold")

        ax = axes.ravel()
        for num, axes in enumerate(tqdm(ax[:len(summary.columns)])):
            sns.barplot(x=summary.index, y=summary[summary.columns[num]], ax=axes, color='m')
            axes.set_xlabel('', fontsize=20, fontweight="bold")
            axes.set_ylabel(summary.columns[num], fontsize=20, fontweight="bold")
            axes.tick_params(axis='both', labelsize=20)
            axes.set_yticklabels(axes.get_yticks(), weight='bold')
            axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            columns = helper_functions.convert_to_subscript(summary.index)
            axes.set_xticklabels(columns, weight='bold', rotation=90)
            axes.spines["bottom"].set_linewidth(3)
            axes.spines["top"].set_linewidth(3)
            axes.spines["left"].set_linewidth(3)
            axes.spines["right"].set_linewidth(3)

        # Remove empty subplots...
        fig = helper_functions.remove_empty_axes(fig)

        plt.get_current_fig_manager().window.showMaximized()
        fig.tight_layout()
        plt.savefig(filepath_to_fuel_properties.parent.joinpath(f"Figures/{num_comp}-component mixtures_summary.tif"), dpi=600)
        plt.close()

# Entropy based analysis...
def entropy_calculation(file_data, filepath_to_fuel_properties, fg_prop_path, n_bins=50):
    """
    Function to calculate entropy of each set of n-component mixtures...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    entropy_value : numpy float
        Entropy of the set of n-component surrogate mixtures.

    """

    filepath_to_fuel_properties = Path(filepath_to_fuel_properties)

    def calculate_entropy(column_data, n_bins=50):
        """
        Function to calculate the entropy of a dataframe of FG distribution of n-component mixtures...

        Parameters
        ----------
        column_data : dataframe
            A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of the n-component category under consideration.
        n_bins : int, optional
            Number of constant bins for entropy calculation. The default is 50.

        Returns
        -------
        entropy_value : numpy float
            Entropy of n-component surrogate mixtures under consideration.

        """

        counts, bin_edges  = np.histogram(column_data, bins=n_bins)
        probabilities = counts/len(column_data)
        entropy_value = entropy(probabilities)

        return entropy_value

    entropy_values = {}
    n_cols = math.ceil(math.sqrt(len(file_data.no_components.unique())))
    n_rows = math.ceil(len(file_data.no_components.unique())/n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 9))
    ax = ax.ravel()
    for num, num_comp in enumerate(file_data.no_components.unique()):
        file_data_modified = file_data[(file_data['no_components'] == num_comp)]
        # Replacing FGs with their code names temporarily...
        fg_ids = pd.read_csv(fg_prop_path, header=0, index_col=None)
        for old_col in file_data_modified.columns[3:]:
            try:
                fg_id = str(fg_ids.loc[fg_ids.UNIFAC_fg[fg_ids.UNIFAC_fg == str(old_col)].index.tolist()[0], 'fg_id'])
                file_data_modified.rename(columns={old_col: fg_id}, inplace=True)
            except: pass
        file_data_modified = file_data_modified.drop_duplicates(subset=file_data_modified.columns[3:3+len([True for col in file_data_modified.columns if str(col).startswith('GUF')])], keep='first')
        # Check for columns with all zero values and remove them...
        zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
        file_data_modified = file_data_modified.drop(columns=zero_columns)

        print(f"\nFor {num_comp}-component mixtures:")
        for col in file_data_modified.iloc[:, 3:].columns:
            if str(col).startswith('GUF'):
                print(f"Entropy of {str(fg_ids.loc[fg_ids.fg_id[fg_ids.fg_id == str(col)].index.tolist()[0], 'UNIFAC_fg'])}: {calculate_entropy(file_data_modified[col])}")
                entropy_values[col] = calculate_entropy(file_data_modified[col], n_bins)

        columns = helper_functions.convert_to_subscript(list(entropy_values.keys()))
        entropy_vals = list(entropy_values.values())

        # Bar graphs...
        ax[num].bar(columns, entropy_vals, color='magenta')
        # ax[num].set_xlabel('Functional groups', fontsize=20, fontweight="bold")
        ax[num].set_ylabel('Entropy', fontsize=20, fontweight="bold")
        ax[num].set_title(f"{num_comp}-component mixtures", fontsize=20, fontweight="bold")
        ax[num].set_xticklabels(columns, weight='bold', rotation=90)
        # Get current y-axis limits...
        y_min, y_max = ax[num].get_ylim()
        # Customize y-axis tick marks if needed...
        ax[num].set_yticks(np.arange(np.floor(y_min), np.ceil(y_max), (y_max-y_min)/5))
        ax[num].tick_params(axis='both', labelsize=20)
        ax[num].set_yticklabels(ax[num].get_yticks(), weight='bold')
        ax[num].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax[num].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[num].set_xticklabels(columns, weight='bold')
        ax[num].spines["bottom"].set_linewidth(3)
        ax[num].spines["top"].set_linewidth(3)
        ax[num].spines["left"].set_linewidth(3)
        ax[num].spines["right"].set_linewidth(3)

    # Remove empty subplots...
    helper_functions.remove_empty_axes(fig)

    plt.get_current_fig_manager().window.showMaximized()
    fig.tight_layout()

    plt.savefig(filepath_to_fuel_properties.parent.joinpath(f"Figures/{num_comp}-component mixtures_entropy.tif"), dpi=600)
    plt.close()

# PCA analysis...
def PCA_analysis(file_data, fg_prop_path):
    """
    Function to determine the suitability of a dataframe for PCA analysis using Kaiser-Meyer-Olkin test...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of the n-component category under consideration.

    Returns
    -------
    None.

    """

    for num_comp in file_data.no_components.unique():
        file_data_modified = file_data[(file_data['no_components'] == num_comp)]
        # Replacing FGs with their code names temporarily...
        fg_ids = pd.read_csv(fg_prop_path, header=0, index_col=None)
        for old_col in file_data_modified.columns[3:]:
            try:
                fg_id = str(fg_ids.loc[fg_ids.UNIFAC_fg[fg_ids.UNIFAC_fg == str(old_col)].index.tolist()[0], 'fg_id'])
                file_data_modified.rename(columns={old_col: fg_id}, inplace=True)
            except: pass
        file_data_modified = file_data_modified.drop_duplicates(subset=file_data_modified.columns[3:3+len([True for col in file_data_modified.columns if str(col).startswith('GUF')])], keep='first')
        # Check for columns with all zero values and remove them...
        zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
        file_data_modified = file_data_modified.drop(columns=zero_columns)

        pca_values = []
        print(f"\nFor {num_comp}-component mixtures:")
        # Check suitablity of using PCA for the dataset...
        # Kaiser-Meyer-Olkin (KMO) Test...
        kmo_all, kmo_model = calculate_kmo(file_data_modified.iloc[:, 3:3+len([True for col in file_data_modified.columns if str(col).startswith('GUF')])])
        print("\nKMO value:", kmo_model)
        pca_values.append(kmo_model)

    if all(x > 0.5 for x in pca_values):
        print("\nPCA analysis maybe suitable...\n")

""" Monte-Carlo analysis of diversity..."""
def calculate_entropy(column_data, n_bins=50):
    """
    Function to calculate the entropy using Monte-Carlo analysis...

    Parameters
    ----------
    column_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of the n-component category under consideration.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    entropy_value : numpy float
        Entropy of n-component surrogate mixtures under consideration.

    """
    counts, bin_edges  = np.histogram(column_data, bins=n_bins)
    probabilities = counts/len(column_data)
    entropy_value = entropy(probabilities)

    return entropy_value

def monte_carlo(data, max_iterations=10, subset_size=10, n_bins=50):
    """
    Function to perform monte-carlo analysis for entropy analysis...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of the n-component category under consideration.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.
    subset_size : int, optional
        The number of mixtures to be considered as a subset. The default is 10.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    entrops: numpy float
        Mean of the entropy for each subset.

    """

    entrops = []
    selected_lists = []
    num_rows = data.shape[0]
    for _ in range(max_iterations):
        entropy_values = {}
        # print(f"{_}/{max_iterations}")
        indices = np.random.choice(num_rows, size=subset_size, replace=False)
        sorted_indices = list(np.sort(indices))
        if sorted_indices not in selected_lists:
            subset = data.iloc[indices]
            selected_lists.append(sorted_indices)
            for col in subset.iloc[:, 3:3+len([True for col in data.columns if str(col).startswith('GUF')])].columns:
                entropy_values[col] = calculate_entropy(subset[col], n_bins)
            entrops.append(np.min(list(entropy_values.values())))
        else:
            _-=1

    return np.mean(entrops)

def average_correlation(file_data, max_iterations, num_comp, fg_prop_path, n_bins=50):
    """
    Function to generate subset categories and perform monte-carlo analysis using multiprocessing...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of the n-component category under consideration.
    max_iterations : int
        Maximum iterations for monte-carlo analysis.
    num_comp : int
        Number of components in the mixtures to be considered.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    desired_num_mix: numpy array
        A numpy array of the subsets.
    avg_corr: numpy array
        A numpy array of the mean entropy for each subset.

    """

    file_data_modified = file_data[(file_data['no_components'] == num_comp)]
    # Replacing FGs with their code names temporarily...
    fg_ids = pd.read_csv(fg_prop_path, header=0, index_col=None)
    for old_col in file_data_modified.columns[3:]:
        try:
            fg_id = str(fg_ids.loc[fg_ids.UNIFAC_fg[fg_ids.UNIFAC_fg == str(old_col)].index.tolist()[0], 'fg_id'])
            file_data_modified.rename(columns={old_col: fg_id}, inplace=True)
        except: pass
    file_data_modified = file_data_modified.drop_duplicates(subset=file_data_modified.columns[3:3+len([True for col in file_data_modified.columns if str(col).startswith('GUF')])], keep='first')
    # Check for columns with all zero values and remove them...
    zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
    file_data_modified = file_data_modified.drop(columns=zero_columns)
    if len(file_data_modified) <= 20:
        desired_num_mix = list(np.arange(4, len(file_data_modified)+1, 2))
    elif len(file_data_modified) > 20 and len(file_data_modified) <= 100:
        desired_num_mix = list(np.arange(4, len(file_data_modified)+1, 5))
    elif len(file_data_modified) > 100 and len(file_data_modified) <= 1000:
        desired_num_mix = list(np.arange(4, 100, 5))+list(np.arange(100, len(file_data_modified)+1, 10))
    elif len(file_data_modified) > 1000 and len(file_data_modified) <= 10000:
        desired_num_mix = list(np.arange(4, 100, 5))+list(np.arange(100, 1000, 10))+list(np.arange(1000, len(file_data_modified)+1, 500))
    else:
        desired_num_mix = list(np.arange(4, 100, 5))+list(np.arange(100, 1000, 10))+list(np.arange(1000, 10500, 500))

    avg_corr = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(monte_carlo, file_data_modified, max_iterations, j, n_bins) for j in desired_num_mix]
        for future in tqdm(as_completed(futures), total=len(desired_num_mix)):
            result = future.result()
            avg_corr.append(result)

    return np.array(desired_num_mix), np.array(avg_corr)

def main(filepath_to_fuel_properties, fg_data, fg_prop_path, max_iterations=100, n_bins=50):
    """
    Function to initialize Monte-Carlo analysis and save necessary figures...

    Parameters
    ----------
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    fg_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 100.
    n_bins : int, optional
        Number of constant bins for entropy calculation. The default is 50.

    Returns
    -------
    None.

    """

    print('\nFinding entropy vs number of mixtures using Monte-Carlo analysis...\n')
    filepath_to_fuel_properties = Path(filepath_to_fuel_properties)
    x_list = []
    y_list = []
    for num_comp in fg_data.no_components.unique():
        x,y = average_correlation(fg_data, max_iterations, num_comp, fg_prop_path, n_bins=50)
        x_list.append(x)
        y_list.append(y)

    df = pd.DataFrame()
    for num, num_comp in enumerate(fg_data.no_components.unique()):
        for j in range(len(x_list[num])):
            df[str(num_comp)+'_mixtures'] = x_list[num][j]
            df[str(num_comp)+'_correlation'] = y_list[num][j]

    df.to_csv(filepath_to_fuel_properties.parent.joinpath("entropy_vs_number_of_mixtures.csv"), index=None)

    # Plotting...
    fig, ax1 = plt.subplots()
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    for r, comp in enumerate(fg_data.no_components.unique()):
        ax1.plot(x_list[r],y_list[r], label=str(comp)+' components', linewidth=3, linestyle='solid')
        ax1.set_xscale('log')
        ax1.set_xlabel('Number of mixtures', fontsize=30, fontweight="bold")
        ax1.set_ylabel('Max correlation', fontsize=30, fontweight="bold")
        ax1.tick_params(axis='both', labelsize=30)
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
        ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_xscale('log')
        ax1.spines["bottom"].set_linewidth(3)
        ax1.spines["top"].set_linewidth(3)
        ax1.spines["left"].set_linewidth(3)
        ax1.spines["right"].set_linewidth(3)
        font = font_manager.FontProperties(size=20, weight='bold')
        ax1.legend(prop=font)

    plt.tight_layout()
    plt.savefig(filepath_to_fuel_properties.parent.joinpath("Figures/entropy_vs_number_of_mixtures.tif"), dpi=600)
    plt.close()