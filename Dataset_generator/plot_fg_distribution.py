# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:33:13 2024
Last Updated: 08/21/2024

@author: Abhinav Abraham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import helper_functions
from tqdm import trange
import math
import os
import warnings
warnings.filterwarnings('ignore')

""" Class for plotting functional group distribution of each n-component mixture... """
class Fuel_Analysis:
    def __init__(self, fuel_name: str, fuel_filepath, fg_prop_path):
        self.fuel_name = fuel_name
        self.fuel_filepath = fuel_filepath
        self.fg_prop_path = fg_prop_path

    def combined_plot_fg_hist(self, df, num_comp: int):
        """
        Function for creating plot of each n-component functional group distribution and save them to drive...

        Parameters
        ----------
        df : dataframe
            A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
        num_comp : int
            The number of component in the mixtures to be considered.

        Returns
        -------
        None.

        """

        fgs = [helper_functions.convert_to_subscript(fg) for fg in df.columns[3:]]

        fuel_data = pd.read_csv(self.fuel_filepath, header=0, index_col=0)
        fg_ids = pd.read_csv(self.fg_prop_path, header=0, index_col=None)
        for old_col in fuel_data.columns[:-2]:
            fg_id = str(fg_ids.loc[fg_ids.fg_id[fg_ids.fg_id == str(old_col)].index.tolist()[0], 'UNIFAC_fg'])
            fuel_data.rename(columns={old_col: fg_id}, inplace=True)

        df = df[df.no_components==num_comp]
        # Convert the functional group distributions of n_comp to a list for easier plotting...
        y = np.array([df.iloc[:, num].to_numpy() for num, i in enumerate(df.columns[3:], start=3)])

        n_cols = math.ceil(math.sqrt(len(df.columns[3:])))
        n_rows = math.ceil(len(df.columns[3:])/n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9))
        plt.get_current_fig_manager().window.showMaximized()
        ax = list(axes.ravel())[:len(y)]
        for h,ax in enumerate(ax):
            # Loop through each subplot and plot the corresponding data...
            if sum(y[h]) != 0:
                sns.distplot(y[h], bins=10, color='green', ax=ax, fit_kws={"color":"white"}, kde=True)
                ax.vlines(x=float(fuel_data.loc[self.fuel_name, str(df.columns[3:][h])].strip("'[]'").split(', ')[0]),
                          ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
                ax.vlines(x=float(fuel_data.loc[self.fuel_name, str(df.columns[3:][h])].strip("'[]'").split(', ')[1]),
                          ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
                ax.set_title(str(fgs[h]), fontsize=20, fontweight="bold")
                ax.set_ylabel('Density', fontsize=20, fontweight="bold")
                ax.set_xlabel('Composition', fontsize=20, fontweight="bold")
                ax.tick_params(axis='both', labelsize=20)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                if any(x > 1 for x in list(ax.get_xticks())):
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                else:
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                if any(y > 1 for y in list(ax.get_yticks())):
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                else:
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.spines["bottom"].set_linewidth(3)
                ax.spines["top"].set_linewidth(3)
                ax.spines["left"].set_linewidth(3)
                ax.spines["right"].set_linewidth(3)
            else:
                ax.vlines(x=float(fuel_data.loc[self.fuel_name, str(df.columns[3:][h])].strip("'[]'").split(', ')[0]),
                          ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
                ax.vlines(x=float(fuel_data.loc[self.fuel_name, str(df.columns[3:][h])].strip("'[]'").split(', ')[1]),
                          ymin=0, ymax=ax.get_ylim()[1], colors='k', linestyles='dashed', linewidth=3, label='Min/Max')
                ax.set_title(str(fgs[h]), fontsize=20, fontweight="bold")
                ax.set_ylabel('Density', fontsize=20, fontweight="bold")
                ax.set_xlabel('Composition', fontsize=20, fontweight="bold")
                ax.tick_params(axis='both', labelsize=20)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                if any(x > 1 for x in list(ax.get_xticks())):
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                else:
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                if any(y > 1 for y in list(ax.get_yticks())):
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                else:
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.spines["bottom"].set_linewidth(3)
                ax.spines["top"].set_linewidth(3)
                ax.spines["left"].set_linewidth(3)
                ax.spines["right"].set_linewidth(3)

        # Remove empty subplots...
        fig = helper_functions.remove_empty_axes(fig)

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.fg_prop_path), f"Figures/densityplot_extended_fg_distributions_{num_comp}_component_mixtures.tif"), dpi=300)
        plt.close()

def main(num_comps_list, fuel_name, fuel_filepath, fg_prop_path, df):
    """
    Funciton to initialize the plotting of FG distributions...

    Parameters
    ----------
    num_comps_list : list
        A list of number of components to be considered when making mixtures.
    fuel_name : str
        Name of target fuel.
    fuel_filepath : str
        Path location for fuel functional group compositions.
    fg_prop_path : str
        Path location for functional group properties.
    df : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.

    Returns
    -------
    None.

    """

    # Initialize class...
    cls_obj = Fuel_Analysis(fuel_name, fuel_filepath, fg_prop_path)

    # Plot combined histrograms for all number of components...
    for i in trange(0,len(num_comps_list)):
        cls_obj.combined_plot_fg_hist(df, num_comps_list[i])