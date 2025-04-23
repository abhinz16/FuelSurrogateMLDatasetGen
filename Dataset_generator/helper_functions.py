# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:42:31 2024
Last Updated: 08/20/2024

@author: Abhinav Abraham
"""

import pandas as pd
import platform
from datetime import date
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
import os
import warnings
warnings.filterwarnings('ignore')

def convert_to_subscript(strings):
    """
    Function to convert a string to superscript or subscript as needed...

    Parameters
    ----------
    strings : str
        String to be corrected.

    Returns
    -------
    converted_strings : str
        Corrected string.

    """

    # Define the subscript translation table...
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    # Convert each string in the list...
    converted_strings = ''.join([s.translate(subscript_map) for s in strings])

    return converted_strings

def combine_fg_data(filenames, fuel_name, fg_prop_path, PC_data, user_name):
    """
    Function to create a pandas dataframe consisting of the functional group distibutions of all the
    surrogate mixtures of all the n-component categories.

    Parameters
    ----------
    filenames : list
        A list of filepaths for each n-component surrogate mixtures.
    fuel_name : str
        Name of the target fuel.
    fg_prop_path : str
        Path to file containing properties of each functional group.
    PC_data : dataframe
        A pandas dataframe of the pure components and its functional group compositions and other details.
    user_name : str
        Provide a user name.

    Returns
    -------
    df : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.

    """

    fg_prop_path = fg_prop_path

    col_df = ['#', 'Fuel', 'no_components']+[str(i) for i in PC_data.columns[0:]]+['Filename', 'Date', 'user_id']
    df = pd.DataFrame(columns=col_df)

    for f in trange(len(filenames)):
        filenames[f] = filenames[f]
        d = date.today()

        if platform.system() == 'Windows':
            plot_details = [k.replace('.', '~').split('~')[0] for l, k in enumerate(filenames[f].split('\\')[-1].split('_')) if l in [2, 6, 12, 14]]
        elif platform.system() == 'Linux':
            plot_details = [k.split('~')[0] for l, k in enumerate(filenames[f].split('/')[-1].split('_')) if l in [2, 6, 12, 14]]
        else:
            raise('Wrong platform. Supports only Windows or Linux.')

        file_data = pd.read_csv(filenames[f], sep=',', header=0)
        for index, hash_number in enumerate(file_data.index):
            data2add = {'#': file_data.iloc[index, 0], 'Fuel': str(plot_details[0]), 'no_components': int(plot_details[-2])}
            for fg_num, fg in enumerate(file_data.columns[2:-2], start=2):
                data2add[fg.split('_')[0].strip()] = float(file_data.iloc[index, fg_num])
            data2add['Filename'] = str(filenames[f])
            data2add['Date'] = str(d.strftime("%m-%d-%Y"))
            data2add['user_id'] = str(user_name)
            df = df.append(data2add, ignore_index=True)

    fg_ids = pd.read_csv(fg_prop_path, header=0, index_col=None)
    for old_col in df.columns[3:-4]:
        fg_id = str(fg_ids.loc[fg_ids.fg_id[fg_ids.fg_id == str(old_col)].index.tolist()[0], 'UNIFAC_fg'])
        df.rename(columns={old_col: fg_id}, inplace=True)

    df2 = df.copy()
    df2.to_csv(os.path.join(os.path.dirname(fg_prop_path),'Output', 'combined_fgs_of_mixtures.csv'), index=None)

    df.drop(columns=['Filename', 'Date', 'user_id'], inplace=True)
    df.fillna(value=0, inplace=True)
    df = df.loc[:, (df != 0).any(axis=0)] # Drop all columns with only zeros...

    return df

def plotting_correlation(filepath_to_fuel_properties, fg_data, x_list, y_list):
    """
    Function to create correlation plots and save them...

    Parameters
    ----------
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    fg_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    x_list : list
        A list of number of mixtures in each subset.
    y_list : list
        Mean of the maximum correlation for each subset.

    Returns
    -------
    None.

    """

    # Plotting...
    fig, ax1 = plt.subplots(figsize=(18, 9))
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    for r, comp in enumerate(fg_data.no_components.unique()):
        ax1.plot(x_list[r],y_list[r], label=str(comp)+' components', linewidth=3, linestyle='solid')
        ax1.set_xscale('log')
        ax1.set_xlabel('Number of mixtures', fontsize=30, fontweight="bold")
        ax1.set_ylabel('Correlation', fontsize=30, fontweight="bold")
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
        ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.tick_params(axis='both', labelsize=30)
        ax1.spines["bottom"].set_linewidth(3)
        ax1.spines["top"].set_linewidth(3)
        ax1.spines["left"].set_linewidth(3)
        ax1.spines["right"].set_linewidth(3)
        font = font_manager.FontProperties(size=20, weight='bold')
        ax1.legend(prop=font)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(filepath_to_fuel_properties), "Figures", "distance_correlation_vs_number_of_mixtures.tif"), dpi=600)
    plt.close()

def remove_empty_axes(fig):
    """
    Function to remove unused subplots from the figure...

    Parameters
    ----------
    fig : matplotlib plot
        Handle for matplotlib plot.

    Returns
    -------
    fig : matplotlib plot
        Return handle for matplotlib plot after removing unused subplots.

    """

    ax2remove = []
    for ax in fig.axes:
        if not ax.has_data():
            fig.delaxes(ax)
            ax2remove.append(ax)

    for ax in fig.axes:
        if ax in ax2remove:
            ax.remove()

    return fig