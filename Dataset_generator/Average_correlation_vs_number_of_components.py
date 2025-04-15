# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:39:45 2024
Last Updated: 03/19/2024

@author: Abhinav Abraham

"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors
import seaborn as sns

#%%
def evaluate_correlation(data):
    corr_matrix_before = data.corr()
    corr_matrix_before.fillna(value=0, inplace=True)
    corr_matrix_before = corr_matrix_before.abs()
    num_columns = len(data.columns)-4
    total_corr = corr_matrix_before.sum().sum() - num_columns  # Subtracting diagonal elements since thery are 1 ...

    return total_corr/(num_columns * (num_columns - 1))

def monte_carlo(data, max_iterations=10, subset_size=10):
    corr = []
    selected_lists = []
    num_rows = data.shape[0]
    for _ in range(max_iterations):
        # print(_)
        indices = np.random.choice(num_rows, size=subset_size, replace=False)
        sorted_indices = list(np.sort(indices))
        if sorted_indices not in selected_lists:
            subset = data.iloc[indices]
            selected_lists.append(sorted_indices)

            corr.append(evaluate_correlation(subset.iloc[:,3:]))
        else:
            _-=1

    return np.mean(corr)

def average_correlation(file_data, max_iterations, num_comp):
    file_data_modified = file_data[(file_data['no_components'] == num_comp) & (file_data['Fuel'] == 'DATASET')]
    print(file_data_modified['no_components'].unique())
    # file_data_modified.drop_duplicates(subset=['Fuel', 'no_components']+list(file_data_modified.columns[3:]), keep='first', inplace=True)
    print(len(file_data_modified))
    if len(file_data_modified) <= 1000:
        desired_num_mix = list(np.arange(10, 100, 5))+list(np.arange(100, len(file_data_modified), 10))
    elif len(file_data_modified) > 1000 and len(file_data_modified) <= 10000:
        desired_num_mix = list(np.arange(10, 100, 5))+list(np.arange(100, 1000, 10))+list(np.arange(1000, len(file_data_modified), 500))
    else:
        desired_num_mix = list(np.arange(10, 100, 5))+list(np.arange(100, 1000, 10))+list(np.arange(1000, 10500, 500))

    avg_corr = []
    for j in desired_num_mix:
        avg_corr.append(monte_carlo(file_data_modified, max_iterations, j))

    return np.array(desired_num_mix), np.array(avg_corr)

def correlation_plot(file_data, num_comp):
    file_data_modified = file_data[(file_data['no_components'] == num_comp)]
    fgs = ['CH\u2083', 'CH\u2082', 'CH', 'C', 'CH\u2082=CH', 'CH=CH', 'CH\u2082=C', 'CH=C', 'ACH',
           'ACCH\u2083', 'ACCH\u2082', 'ACCH']

    file_data_modified.drop(columns=['Fuel', '#', 'no_components'], inplace=True)
    corr = file_data_modified.corr(method = 'pearson', min_periods = file_data_modified.shape[0]).round(3)
    corr.fillna(value=0, inplace=True)

    print(evaluate_correlation(file_data_modified.iloc[:,:]))

    # # # Plotting...
    plt.figure()
    ax = sns.heatmap(corr, annot=True, cmap="PiYG", linecolor='k', linewidths=1, cbar=True,
                      annot_kws={"size": 22, 'weight':'bold', 'color':'black'}, vmin=-1, vmax=1)
    ax.set_xticklabels(fgs, ha='center', fontsize=30, fontweight="bold", rotation=90)
    ax.set_yticklabels(fgs, va='center', fontsize=30, fontweight="bold")
    ax.tick_params(left=False, bottom=False)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.tick_params(axis='both', which='both', length=0)
    cbar.ax.set_yticklabels(cbar.ax.get_yticks(), weight='bold')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#%%
file_data = pd.read_csv(r"Z:\PhD\Surrogate development\Original_surrogate_model\Extended_mixtures_dataset_only_DATASET_4.csv", header=0)
max_iterations = 2000

x_list = []
y_list = []
for num_comp in [4,5,6,7,8]:
    x,y = average_correlation(file_data, max_iterations, num_comp)
    x_list.append(x)
    y_list.append(y)

#%%
# colors = list(mcolors.CSS4_COLORS.keys())
colors=['r', 'g', 'k', 'b', 'm']
linestyles = ['solid', 'dashed', '-.', ':', 'dotted']
fig, ax1 = plt.subplots()
left, bottom, width, height = [0.3, 0.42, 0.45, 0.45]
# ax2 = fig.add_axes([left, bottom, width, height])

for r in range(5):
    # chosen_color = np.random.choice(colors)
    # chosen_linestyle = np.random.choice(linestyles)
    # colors.remove(chosen_color)
    # linestyles.remove(chosen_linestyle)
    ax1.plot(x_list[r],y_list[r]*100,color=colors[r], label=str(r+4)+' components', linewidth=4, linestyle=linestyles[r])
    # ax1.scatter(x,y,s=150,c=chosen_color, label=str(num_comp)+' components', marker='o')
    ax1.set_xlabel('Number of mixtures', fontsize=40, fontweight="bold")
    ax1.set_ylabel('Mean Correlation (%)', fontsize=40, fontweight="bold")
    ax1.tick_params(axis='both', labelsize=40)
    ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
    ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_xscale('log')
    ax1.spines["bottom"].set_linewidth(3)
    ax1.spines["top"].set_linewidth(3)
    ax1.spines["left"].set_linewidth(3)
    ax1.spines["right"].set_linewidth(3)
    # ax1.set_xlim(-150, 8000)
    font = font_manager.FontProperties(size=40, weight='bold')
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

#%%
file_data = pd.read_csv(r"Z:\PhD\Surrogate development\Original_surrogate_model\Extended_mixtures_dataset_only_DATASET_4.csv", header=0)
correlation_plot(file_data ,6)
