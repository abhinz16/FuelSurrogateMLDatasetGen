# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:00:05 2024
Last Updated: 08/21/2024

@author: Abhinav Abraham
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings('ignore')

def get_diff(filepath_to_fuel_properties, correlation, entropy):
    """
    Function to find the difference between correlations and entropies for each subset and save the plot showing the difference...

    Parameters
    ----------
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    correlation : numpy array
        A numpy array of the mean correlation for each subset.
    entropy : numpy array
        A numpy array of the mean entropy for each subset.

    Returns
    -------
    intersect : boolean
        True or False. Does the correlations and entropies interset at any point.

    """

    correlation_data = correlation['Mean maximum correlation'].to_numpy()
    correlation_data = np.array([i if i<=1 else correlation_data[num-1] for num, i in enumerate(correlation_data)])
    no_of_mixtures_correlation = correlation['Number of mixtures'].to_numpy()
    paired_array = list(zip(no_of_mixtures_correlation, correlation_data))
    paired_array.sort()
    no_of_mixtures_correlation = np.array([pair[0] for pair in paired_array])
    correlation_data = np.array([pair[1] for pair in paired_array])

    entropy_data = entropy['Mean minimum entropy'].to_numpy()
    no_of_mixtures_entropy = entropy['Number of mixtures'].to_numpy()
    paired_array = list(zip(no_of_mixtures_entropy, entropy_data))
    paired_array.sort()
    no_of_mixtures_entropy = np.array([pair[0] for pair in paired_array])
    entropy_data = np.array([pair[1] for pair in paired_array])

    # Find difference...
    intersect = False
    for k in range(len(correlation_data)):
        if np.round((correlation_data[k]),2) - np.round((entropy_data[k]),2) == 0:
            intersect = True
            break
        else:
            pass

    diff = np.abs(correlation_data - entropy_data)

    # Plotting...
    fig, ax = plt.subplots()
    plt.get_current_fig_manager().window.showMaximized()
    ax.scatter(no_of_mixtures_entropy, diff, color='r', marker='o', s=50)
    ax.set_xlabel('Number of mixtures', fontsize=30, fontweight="bold")
    ax.set_ylabel('Correlation-Entropy', fontsize=30, fontweight="bold")
    ax.set_xticklabels(ax.get_xticks(), ha='center', fontsize=30, fontweight="bold")
    ax.set_yticklabels(ax.get_yticks(), va='center', fontsize=30, fontweight="bold")
    ax.tick_params(left=False, bottom=False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=30)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.pause(1)

    plt.tight_layout()
    plt.savefig(Path(filepath_to_fuel_properties).parent.joinpath("Figures/Optimal_number_of_mixtures.tif"), dpi=600)

    return intersect

def main(filepath_to_fuel_properties, correlation, entropy, threshold=0.1, intersect=False):
    """
    Function to find the minimum number of mixtures for optimal dataset...

    Parameters
    ----------
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    correlation : numpy array
        A numpy array of the mean correlation for each subset.
    entropy : numpy array
        A numpy array of the mean entropy for each subset.
    threshold : float, optional
        Threshold for finding minimum number of mixtures for optimal dataset. The default is 0.1.
    intersect : boolean, optional
        True or False. Does the correlations and entropies interset at any point. The default is False.

    Returns
    -------
    minimum_no_of_mixtures : int
        The minimum number of mixtures for an optimal dataset.

    """

    correlation_data = correlation['Mean maximum correlation'].to_numpy()
    correlation_data = np.array([i if i<=1 else correlation_data[num-1] for num, i in enumerate(correlation_data)])
    no_of_mixtures_correlation = correlation['Number of mixtures'].to_numpy()
    paired_array = list(zip(no_of_mixtures_correlation, correlation_data))
    paired_array.sort()
    no_of_mixtures_correlation = np.array([pair[0] for pair in paired_array])
    correlation_data = np.array([pair[1] for pair in paired_array])

    entropy_data = entropy['Mean minimum entropy'].to_numpy()
    no_of_mixtures_entropy = entropy['Number of mixtures'].to_numpy()
    paired_array = list(zip(no_of_mixtures_entropy, entropy_data))
    paired_array.sort()
    no_of_mixtures_entropy = np.array([pair[0] for pair in paired_array])
    entropy_data = np.array([pair[1] for pair in paired_array])

    if not intersect:
        print('\nNo intersection found...')
        diff = np.abs(correlation_data - entropy_data)
        minimum_no_of_mixtures = no_of_mixtures_entropy[np.where(diff<=threshold)[0][0]]
        print(f"Minimum number of mixtures: {minimum_no_of_mixtures}")
    else:
        print('\nIntersection found...')
        diff = np.abs(correlation_data - entropy_data)
        for i in range(len(diff) - 1, -1, -1):
            if diff[i] <= threshold:
                minimum_no_of_mixtures = no_of_mixtures_entropy[i]
                print(f"Minimum number of mixtures: {minimum_no_of_mixtures}")
                break

        # # Define the points (first and last point of the curve)
        # p1 = np.array([no_of_mixtures_entropy[0], entropy_data[0]])  # First point
        # p2 = np.array([no_of_mixtures_entropy[-1], entropy_data[-1]])  # Last point

        # # Calculate the distance of each point to the line connecting p1 and p2
        # line_vec = p2 - p1  # Vector representing the line
        # line_vec_norm = line_vec / np.linalg.norm(line_vec)  # Normalized line vector

        # # Compute the vector from p1 to each point on the curve
        # point_vecs = np.vstack([no_of_mixtures_entropy, entropy_data]).T - p1

        # # Project the point vectors onto the line vector and find the perpendicular distance
        # proj_line = np.dot(point_vecs, line_vec_norm)
        # proj_point = np.outer(proj_line, line_vec_norm) + p1
        # perpendicular_vecs = point_vecs - proj_point

        # # Compute distances (L2 norm)
        # distances = np.linalg.norm(perpendicular_vecs, axis=1)
        # plt.figure()
        # plt.plot(distances)

        # # Find the index of the maximum distance (the elbow point)
        # elbow_index = np.argmin(distances)

        # minimum_no_of_mixtures = no_of_mixtures_entropy[elbow_index]
        # print(f"Minimum number of mixtures: {minimum_no_of_mixtures}")

    # Plotting...
    fig, ax = plt.subplots()
    plt.get_current_fig_manager().window.showMaximized()
    ax.scatter(no_of_mixtures_entropy, correlation_data, color='r', marker='o', s=50, label='Correlation')
    ax.scatter(no_of_mixtures_entropy, entropy_data, color='k', marker='o', s=50, label='Entropy')
    ymin = min(ax.get_yticks())
    ymax = max(ax.get_yticks())
    ax.vlines(minimum_no_of_mixtures, ymin=ymin, ymax=ymax, linestyles='dashed', linewidth=5, colors='g', label='Minimum')
    ax.set_xlabel('Number of mixtures', fontsize=30, fontweight="bold")
    ax.set_ylabel('Correlation/Entropy', fontsize=30, fontweight="bold")
    ax.set_ylim(ymin, 1)
    ax.set_xticklabels(ax.get_xticks(), ha='center', fontsize=30, fontweight="bold")
    ax.set_yticklabels(ax.get_yticks(), va='center', fontsize=30, fontweight="bold")
    ax.tick_params(left=False, bottom=False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=30)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(Path(filepath_to_fuel_properties).parent.joinpath("Figures/Optimal_number_of_mixtures.tif"), dpi=600)
    plt.close()

    return minimum_no_of_mixtures