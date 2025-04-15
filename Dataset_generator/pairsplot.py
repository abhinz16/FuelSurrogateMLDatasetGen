# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:39:45 2024
Last Updated: 07/02/2024

@author: Abhinav Abraham

"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def plot_pairplots(file_data, num_comp, filepath_to_fuel_properties, color_palette='Spectral'):
    """
    Function to generate pairsplots of the functional group distributions of the generated surrogate mixtures of each n-component category...

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    num_comp : int
        n-component category to be considered.
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    color_palette : str, optional
        The default is 'Spectral'.

    Returns
    -------
    None.

    """
    filepath_to_fuel_properties = Path(filepath_to_fuel_properties)
    file_data_modified = file_data.drop_duplicates(subset=file_data.columns[3:], keep='first')
    zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
    file_data_modified = file_data_modified.drop(columns=zero_columns)
    df_cols = file_data_modified.iloc[:, 3:].columns
    df_cols = [text.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")) for text in df_cols]
    sns.set_palette(color_palette)
    g = sns.pairplot(file_data_modified.iloc[:, 3:], diag_kind='kde', plot_kws={'s': 100}, corner=False)
    for num, ax in enumerate(g.axes.flatten()):
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontweight('bold')
        ax.tick_params(axis='both', which='both', labelsize=15, width=0, length=0)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.xaxis.label.set_size(17)
        ax.yaxis.label.set_size(17)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)
        if ax.xaxis.label.get_text() == 'DCN':
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        if ax.yaxis.label.get_text() == 'DCN':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if ax.xaxis.label.get_text() == 'Molecular Weight':
            ax.set_xlabel('MW')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        if ax.yaxis.label.get_text() == 'Molecular Weight':
            ax.set_ylabel('MW')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if ax.get_xlabel() in file_data_modified.iloc[:, 3:].columns:  # Check if x label corresponds to a column name in the DataFrame
            ax.set_xlabel(df_cols[file_data_modified.iloc[:, 3:].columns.get_loc(ax.get_xlabel())])
        if ax.get_ylabel() in file_data_modified.iloc[:, 3:].columns:
            ax.set_ylabel(df_cols[file_data_modified.iloc[:, 3:].columns.get_loc(ax.get_ylabel())])
    
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.savefig(filepath_to_fuel_properties.parent.joinpath(f"Figures/{num_comp}_component_mixtures_pairplot.tif"), dpi=600)
    plt.close()

def main(data, filepath_to_fuel_properties):
    """
    Function to initiate the generation of pairplots for desired n-component categories...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.

    Returns
    -------
    None.

    """
    for num_comp in tqdm(data.no_components.unique()):
        plot_pairplots(data, num_comp, filepath_to_fuel_properties, color_palette='RdYlBu')