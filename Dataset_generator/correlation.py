# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:39:45 2024
Last Updated: 08/21/2024

@author: Abhinav Abraham
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import dcor
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

def distance_corr_dataframe(data, min_periods: int = 1):
    """
    Function to find the distance correlation.

    Parameters
    ----------
    data : dataframe
        Normalized pandas dataframe of the surrogate mixtures with its functional group compositions.
    min_periods : int, optional
        The default is 1.

    Returns
    -------
    data_dist_corr : dataframe
        A pandas dataframe of the distance correlation between each variables (funcional groups).

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
    Function to normalize the dataframe with the functional group distibutions...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe of the surrogate mixtures with its functional group compositions.

    Returns
    -------
    data_corr : dataframe
        Normalized pandas dataframe of the surrogate mixtures with its functional group compositions.

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
    Function to find the maximum distance correlation between the variables...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe of the distance correlation between each variables (funcional groups).

    Returns
    -------
    data_max : float
        Maximum correlation value between any two variables.
    absolute_max_index_row : int
        Index row position of the maximum correlation value in the correlation dataframe.
    absolute_max_index_col : int
        Index column position of the maximum correlation value in the correlation dataframe.
    correlation_type : str
        Dirtection of correlation. This vill always be positive for distanc correlation
    original_correlation_value : float
        Correlation value between two variables.

    """
    data_corr = evaluate_correlation(data)
    data_max = data_corr.abs().values.max()
    absolute_max_index = data_corr.abs().values.argmax()
    absolute_max_index_row, absolute_max_index_col = divmod(absolute_max_index, len(data_corr))

    # Get the sign of the correlation coefficient at the index...
    original_correlation_value = data_corr.iloc[absolute_max_index_row, absolute_max_index_col]
    if original_correlation_value < 0:
        correlation_type = "negatively"
    elif original_correlation_value > 0:
        correlation_type = "positively"
    else:
        correlation_type = "not correlated"

    return (data_max, absolute_max_index_row, absolute_max_index_col, correlation_type, original_correlation_value)

def monte_carlo(data, max_iterations=10, subset_size=10):
    """
    Monte-Carlo analysis for determining the mean of highest correlation between two functional groups for a
    subset of size 'subset_size' for a total iterations of 'max_iterations'...

    Parameters
    ----------
    data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.
    subset_size : int, optional
        The number of mixtures to be considered as a subset. The default is 10.

    Returns
    -------
    A tuple of the mean of the maximum correaltion for each subset along with its indices and correlation direction.

    """
    maxi = []
    maxi_indices = []
    selected_lists = []
    correlation_values = []
    num_rows = data.shape[0]

    for _ in range(max_iterations):
        # print(_)
        indices = np.random.choice(num_rows, size=subset_size, replace=False)
        sorted_indices = list(np.sort(indices))
        if sorted_indices not in selected_lists:
            subset = data.iloc[indices]
            selected_lists.append(sorted_indices)
            maxims = evaluate_maximum(subset.iloc[:,3:-1])

            maxi.append(maxims[0])
            maxi_indices.append((maxims[1], maxims[2], maxims[3]))
            correlation_values.append(maxims[4])
        else:
            _-=1

    # Find the maximum occurring indices...
    max_occuring_indices = max(Counter(maxi_indices), key=Counter(maxi_indices).get)
    # Find the corresponding original correlation value...
    max_corr_value_index = maxi_indices.index(max_occuring_indices)

    return (np.mean(maxi), max_occuring_indices, correlation_values[max_corr_value_index])

def average_correlation(file_data, max_iterations=10, num_comp=4):
    """
    Function for multiprocessed monte-carlo correlation analysis of each subset to find its mean of maximum correlation.

    Parameters
    ----------
    file_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 10.
    num_comp : int, optional
        n-component category to consider. The default is 4.

    Returns
    -------
    desired_num_mix: Numpy array
        Array of subsets.
    avg_corr: Numpy array
        Array of average correlation foe each subset.
    max_correlated_fgs_w_type : str
        Gives the functional groups that are highly corelated for each n-component category along with the correlation direction.
    max_corr_value : float
        Gives the maximum correlation for each n-component category.

    """
    file_data_modified = file_data[(file_data['no_components'] == num_comp)]
    file_data_modified = file_data_modified.drop_duplicates(subset=file_data_modified.columns[3:], keep='first')
    zero_columns = file_data_modified.iloc[:, 3:].columns[file_data_modified.iloc[:, 3:].eq(0).all()]
    file_data_modified = file_data_modified.drop(columns=zero_columns)
    if len(file_data_modified) <= 20:
        desired_num_mix = list(np.arange(10, len(file_data_modified)+1, 2))
    elif len(file_data_modified) > 20 and len(file_data_modified) <= 100:
        desired_num_mix = list(np.arange(10, len(file_data_modified)+1, 5))
    elif len(file_data_modified) > 100 and len(file_data_modified) <= 1000:
        desired_num_mix = list(np.arange(10, 100, 5))+list(np.arange(100, len(file_data_modified)+1, 10))
    elif len(file_data_modified) > 1000 and len(file_data_modified) <= 10000:
        desired_num_mix = list(np.arange(10, 100, 5))+list(np.arange(100, 1000, 10))+list(np.arange(1000, len(file_data_modified)+1, 500))
    else:
        desired_num_mix = list(np.arange(10, 100, 5))+list(np.arange(100, 1000, 10))+list(np.arange(1000, 10500, 500))

    avg_corr = []
    max_correlated_fgs_w_type = []
    max_corr_value = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(monte_carlo, file_data_modified, max_iterations, j) for j in desired_num_mix]
        for future in tqdm(as_completed(futures), total=len(desired_num_mix)):
            result = future.result()
            avg_corr.append(result[0])
            max_correlated_fgs_w_type.append(result[1])
            max_corr_value.append(result[2])

    return np.array(desired_num_mix), np.array(avg_corr), max_correlated_fgs_w_type, max_corr_value

def main(filepath_to_fuel_properties, fg_data, max_iterations=100):
    """
    Function to initiate the monte-carlo correlation analysis...

    Parameters
    ----------
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    fg_data : dataframe
        A pandas dataframe consisting of the functional group distibutions of all the surrogate mixtures of all the n-component categories.
    max_iterations : int, optional
        Maximum iterations for monte-carlo analysis. The default is 100.

    Returns
    -------
    x_list : list
        A list of number of mixtures in each subset.
    y_list : list
        Mean of the maximum correlation for each subset.

    """
    x_list = []
    y_list = []
    for num_comp in fg_data.no_components.unique():
        x,y,most_corr_fgs,max_corr_value = average_correlation(fg_data, max_iterations, num_comp)
        x_list.append(x)
        y_list.append(y)

    df = pd.DataFrame()
    for num, num_comp in enumerate(fg_data.no_components.unique()):
        for j in range(len(x_list[num])):
            df[str(num_comp)+'_mixtures'] = x_list[num][j]
            df[str(num_comp)+'_correlation'] = y_list[num][j]

    df.to_csv(os.path.join(os.path.dirname(filepath_to_fuel_properties), "Output", "distance_correlation_vs_number_of_mixtures.csv"), index=None)

    return x_list, y_list