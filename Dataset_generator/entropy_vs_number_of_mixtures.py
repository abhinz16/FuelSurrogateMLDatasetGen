# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:39:45 2024
Last Updated: 06/23/2024

@author: Abhinav Abraham

"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

#%%
# Entropy based analysis...
def calculate_entropy(column_data):
    counts, bin_edges  = np.histogram(column_data, bins=30)
    probabilities = counts/len(column_data)

    entropy_value = entropy(probabilities)

    return entropy_value

def monte_carlo(data, max_iterations=10, subset_size=10):
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
            for col in subset.iloc[:, 3:-2].columns:
                entropy_values[col] = calculate_entropy(subset[col])
            entrops.append(np.min(list(entropy_values.values())))
        else:
            _-=1

    return np.mean(entrops)

def average_correlation(file_data, max_iterations, num_comp):
    file_data_modified = file_data[(file_data['no_components'] == num_comp) & (file_data['Fuel'] == 'DATASET')]
    file_data_modified = file_data_modified.drop_duplicates(subset=file_data_modified.columns[3:-2], keep='first')
    # Check for columns with all zero values and remove them...
    zero_columns = file_data_modified.iloc[:, 3:-2].columns[file_data_modified.iloc[:, 3:-2].eq(0).all()]
    file_data_modified = file_data_modified.drop(columns=zero_columns)
    # file_data_modified = file_data_modified.iloc[:100, :]
    # print(file_data_modified['no_components'].unique())
    # file_data_modified.drop_duplicates(subset=['Fuel', 'no_components']+list(file_data_modified.columns[3:]), keep='first', inplace=True)
    print()
    print(f"\nLength of {num_comp}-component mixtures: {len(file_data_modified)}")
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
        futures = [executor.submit(monte_carlo, file_data_modified, max_iterations, j) for j in desired_num_mix]
        for future in tqdm(as_completed(futures), total=len(desired_num_mix)):
            result = future.result()
            avg_corr.append(result)

    return np.array(desired_num_mix), np.array(avg_corr)

#%%
filepath = "/media/NAS/akayam2/PhD/Surrogate development/Surrogate_mixtures/Surrogate_generation_1/Surrogate_1_only_DATASET_21_PCs.csv"
# filepath = r"Z:\PhD\Surrogate development\Surrogate_mixtures\Surrogate_generation_1\Surrogate_1_only_DATASET_21_PCs.csv"
file_data = pd.read_csv(filepath, header=0)
max_iterations = 10000

x_list = []
y_list = []
for num_comp in [2,3,4,5,6,7,8]:
    x,y = average_correlation(file_data, max_iterations, num_comp)
    x_list.append(x)
    y_list.append(y)

#%%
dfff = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'])

data2append = {'a': x_list[0], 'b': y_list[0],
                'c': x_list[1], 'd': y_list[1],
                'e': x_list[2], 'f': y_list[2],
                'g': x_list[3], 'h': y_list[3],
                'i': x_list[4], 'j': y_list[4],
                'k': x_list[5], 'l': y_list[5],
                'm': x_list[6], 'n': y_list[6],}

dfff = dfff.append(data2append, ignore_index=True)

read_df = dfff.copy()
dfff = pd.DataFrame(index=range(500), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'])
for col in read_df.columns:
    d = []
    for u in read_df.loc[0,str(col)]:
        if u not in ['[', ']', ' ']:
            d.append(float(u))
    dfff[str(col)][:len(d)] = d

dfff.to_csv("/media/NAS/akayam2/PhD/Surrogate development/Surrogate_mixtures/temp_entropy.csv", index=None)

#%%
# colors = list(mcolors.CSS4_COLORS.keys())
colors=['r', 'g', 'k', 'b', 'm', 'c', 'darkorange', 'blueviolet']
linestyles = ['dashed']
fig, ax1 = plt.subplots()
left, bottom, width, height = [0.3, 0.42, 0.45, 0.45]
# ax2 = fig.add_axes([left, bottom, width, height])

for r in range(7):
    # chosen_color = np.random.choice(colors)
    # chosen_linestyle = np.random.choice(linestyles)
    # colors.remove(chosen_color)
    # linestyles.remove(chosen_linestyle)
    ax1.plot(x_list[r],y_list[r],color=colors[r], label=str(r+2)+' components', linewidth=4, linestyle=linestyles[0])
    # ax1.scatter(x,y,s=150,c=chosen_color, label=str(num_comp)+' components', marker='o')
    ax1.set_xlabel('Number of mixtures', fontsize=30, fontweight="bold")
    ax1.set_ylabel('Minimum Entropy', fontsize=30, fontweight="bold")
    ax1.tick_params(axis='both', labelsize=30)
    ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
    ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis='y', style='sci', useMathText=True)
    ax1.yaxis.offsetText.set_fontweight('bold')
    ax1.yaxis.offsetText.set_fontsize(20)
    ax1.set_xscale('log')
    ax1.spines["bottom"].set_linewidth(3)
    ax1.spines["top"].set_linewidth(3)
    ax1.spines["left"].set_linewidth(3)
    ax1.spines["right"].set_linewidth(3)
    # ax1.set_xlim(-150, 8000)
    font = font_manager.FontProperties(size=20, weight='bold')
    ax1.legend(prop=font)

    # ax2.plot(x_list[r][:np.where(x_list[r]>=60)[0][0]],y_list[r][:np.where(x_list[r]>=60)[0][0]]*100,color=chosen_color, label=str(num_comp)+' components', linewidth=3, linestyle=chosen_linestyle)
    # # ax2.scatter(x[:np.where(x>=450)[0][0]],y[:np.where(x>=450)[0][0]],s=150,c=chosen_color, label=str(num_comp)+' components', marker='o')
    # ax2.set_xlabel('Number of mixtures', fontsize=20, fontweight="bold")
    # ax2.set_ylabel('Average Correlation', fontsize=20, fontweight="bold")
    # ax2.tick_params(axis='both', labelsize=20)
    # ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    # ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    # ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # # ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax2.spines["bottom"].set_linewidth(3)
    # ax2.spines["top"].set_linewidth(3)
    # ax2.spines["left"].set_linewidth(3)
    # ax2.spines["right"].set_linewidth(3)
    # ax2.legend(prop=font)

# plt.get_current_fig_manager().window.showMaximized()
# plt.tight_layout()