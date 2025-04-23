# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:26:48 2025

@author: Abhinav Abraham

"""

#%%
"""
folderpath - Provide the location to store the data. This location should include all the files in 'Dataset_generator' folder.
user_name - Provide user name here.
num_components - Number of components to be in each surrogate mixture. Provide number of components in a list.
threshold - Provide the threshold below which the component is considered as trace component.
DCN_constr - Provide the DCN constraints. True for yes and False for no.
MW_constr - Provide the Molecular Weight constraints. True for yes and False for no.
max_iter - Provide the maximum number of iterations to search for potential surrogates of each n-component category.
max_mixtures - Provide the number of mixtures that the prgram should try to find for each n-component category mixtures.
n_bins - Number of constant bins for entropy calculations. Give 50, if unsure.
max_iterations_each - Correlation and entropy monte-carlo analysis for each n-component category.
max_iterations_full - Correlation and entropy monte-carlo analysis irrespective of n-component category.
max_iterations_optimal - Maximum number of iterations to find a subset that is the optimal dataset.

"""

CONFIG = {
    "folderpath": '/Dataset_generator',
    "user_name": '',
    "num_components": [1,2,3,4,5,6],
    "threshold": 0.01,
    "DCN_constr": False,
    "MW_constr": False,
    "max_iter": 1000,
    "max_mixtures": 50,
    "n_bins": 50,
    "max_iterations_each": 500,
    "max_iterations_full": 500,
    "max_iterations_optimal": 500,
}
