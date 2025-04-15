# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:19:51 2024
Last Updated: 03/29/2025

@author: Abhinav Abraham

"""

from pathlib import Path
import pandas as pd
import numpy as np
import random
from scipy.optimize import minimize
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

global num_FGs

def read_component_db(PC_data, DCN_constr, MW_constr, family_restriction):
    """
    Function for loading up the pure component database.

    Parameters
    ----------
    PC_data : dataframe with FG quantity of components...
        path to the file containing the pure component properties

    Returns
    -------
    group_MWs: a numpy array, molecular weight of each functional group

    components: a list of class pure components

    """

    global num_FGs
    group_MWs = np.zeros(num_FGs)
    components = []
    for j in range(PC_data.shape[0]):
        groups_comp = np.zeros(num_FGs)
        for i,col in enumerate(PC_data.columns[:num_FGs]):
            if j==0:
                group_MWs[i]=PC_data.loc['fg_wt', col]
            else:
                groups_comp[i]=PC_data.iloc[j,i]
        if j>=1:
            name = PC_data.index[j]
            if DCN_constr:
                try:
                    components.append(pure_component(name,groups_comp,PC_data.loc[name,'DCN'],PC_data.loc[name,'Molecular Weight'],PC_data.loc[name,'TYPE'],PC_data.loc[name,'density']))
                except:
                    raise ValueError('/nDCN for pure components not found. Provide DCN data for all the pure components in the component information file...\n')
            else:
                if 'density' in PC_data.columns:
                    components.append(pure_component(name,groups_comp,PC_data.loc[name,'Molecular Weight'],PC_data.loc[name,'density']))
                elif 'TYPE' in PC_data.columns or family_restriction==True:
                    try:
                        components.append(pure_component(name,groups_comp,PC_data.loc[name,'Molecular Weight'],PC_data.loc[name,'TYPE']))
                    except:
                        if family_restriction:
                            raise ValueError('/nFamily type for pure components not found. Provide family type data for all the pure components in the component information file...\n')
                        else: pass
                elif 'density' in PC_data.columns and 'TYPE' in PC_data.columns:
                    components.append(pure_component(name,groups_comp,PC_data.loc[name,'Molecular Weight'],PC_data.loc[name,'TYPE'],PC_data.loc[name,'density']))
                else:
                    components.append(pure_component(name,groups_comp,'N/A',PC_data.loc[name,'Molecular Weight']))

    return(group_MWs,components)

def read_fuels_db(filepath_to_fuel_properties, PC_data, DCN_constr, MW_constr):
    """
    Function for loading up the target fragementation database.
    This function is a bit garbage because this is an odd file to parse
    Parameters
    ----------
    filepath_to_fuel_properties : string or path
        path to the file containing the fuel database

    Returns
    -------
    fuels: a list of class fuel_targets

    """

    global num_FGs
    fuel_data= pd.read_csv(filepath_to_fuel_properties, header=0, index_col=0)
    fuels = []
    for j in range(fuel_data.shape[0]):
        name = fuel_data.index[j]
        fg_maxs = np.zeros(num_FGs)
        fg_mins = np.zeros(num_FGs)
        for i,col in enumerate(PC_data.columns[:num_FGs]):
            fg_mins[i]=float(fuel_data.loc[name,col].strip('[]').split(',')[0])
            fg_maxs[i]=float(fuel_data.loc[name,col].strip('[]').split(',')[1])

        if DCN_constr:
            dcn_min=float(fuel_data.loc[name,'DCN'].strip('[]').split(',')[0])
            dcn_max=float(fuel_data.loc[name,'DCN'].strip('[]').split(',')[1])
        else:
            dcn_min = 'N/A'
            dcn_max = 'N/A'
        if MW_constr:
            try:
                MW_min=float(fuel_data.loc[name,'Molecular Weight'].strip('[]').split(',')[0])
                MW_max=float(fuel_data.loc[name,'Molecular Weight'].strip('[]').split(',')[1])
            except:
                MW_min=float(fuel_data.loc[name,'Molecular Weight'])-0.5
                MW_max=float(fuel_data.loc[name,'Molecular Weight'])+0.5
        else:
            MW_min = 'N/A'
            MW_max = 'N/A'

        fuels.append(fuel_target(name,fg_mins,fg_maxs,(dcn_min,dcn_max),(MW_min,MW_max),))

    return(fuels)

class fuel_target():
    """class for fuel surrogate targets"""
    def __init__(self,name,fg_mins,fg_maxs,DCN=('N/A','N/A'),MW=(0.0,200),dens=('N/A','N/A')):
        """
        Parameters
        ----------
        name : string, optional
            the name of the fuel. The default is None.
        fg_mins : np.array of floats, optional
            minimum bounds of functional groups. The default is np.zeros(num_FGs).
        fg_maxs : np.array of floats, optional
            maximum bounds of functional groups. The default is np.ones(num_FGs).
        DCN : tuple of floats, optional
            DCN limits of fuel. The default is (0.0,100).
        MW : tuple of floats, optional
            molecular weight limits of fuel. The default is (0.0,200).
        dens : tuple of floats, optional
            density limits of fuel. The default is (0.0,1)

        Returns
        -------
        None.

        """

        self.name=name
        self.fg_mins=fg_mins
        self.fg_maxs=fg_maxs
        self.DCN=DCN
        self.MW=MW
        self.dens=dens

    def __repr__(self):
        return(f'Fuel target: {self.name} with MW= {self.MW}, DCN= {self.DCN}, density= {self.dens}\n')


class pure_component():
    """class for pure components"""
    def __init__(self,name,fgs,DCN='N/A',MW=0.0,mol_type='N/A',dens='N/A'):
        """

        Parameters
        ----------
        name : STRING
            name of the pure component.
        fgs : 1d numpy array of ints
            a numpy array of integers representing fgs.
        DCN : float
            DCN of the pure component.
        MW : float
            molecular weight of pure component in g/mol.
        mol_type : string
            family of molecule: currently parrafins, iso, cyclics, aromatics.
        dens : float
            pure component density in g/cm3.

        Returns
        -------
        None.

        """

        self.name=name
        self.fgs=fgs
        self.DCN=DCN
        self.MW=MW
        self.mol_type=mol_type
        self.dens=dens

    def __repr__(self):
        return(f'Component: {self.name} with MW= {self.MW}, DCN= {self.DCN}, density= {self.dens}\n')


class mixture():
    """class for a mixture"""
    def __init__(self, group_MWs):
        self.mix=[]
        self.mol_frac=np.array([])
        self.group_MWs = group_MWs

    def add_component(self,x,pure_component):
        self.mix.append(pure_component)
        self.mol_frac= np.append(self.mol_frac, x)

    def update_mol_frac(self,x):
        self.mol_frac=x

    def calc_MW(self):
        MW=0.0
        for i,comp in enumerate(self.mix):
            MW+=comp.MW*self.mol_frac[i]
        return(MW)

    def calc_Y(self):
        y=np.zeros_like(self.mol_frac)
        MW_mix = self.calc_MW()
        for i, comp in enumerate(self.mix):
            y[i]=self.mol_frac[i]*comp.MW / MW_mix
        return(y)

    def calc_dens(self):
        #Assume ideal mixtures...
        vol = 0.0
        for i,comp in enumerate(self.mix):
            vol += self.mol_frac[i]/comp.dens
        return(1/vol)

    def calc_DCN(self):
        #Assume ideal mixtures...
        DCN_mix = 0.0
        for i,comp in enumerate(self.mix):
            DCN_mix += self.mol_frac[i]/comp.dens*comp.DCN
        return(DCN_mix)

    def fragment(self):
        y_fg=np.zeros(num_FGs)
        MW_mix=self.calc_MW()
        for i,comp in enumerate(self.mix):
            y_fg += self.mol_frac[i]*(comp.fgs*self.group_MWs)
        y_fg/=MW_mix
        return(y_fg)

    def __repr__(self):
        string='A mixture containing:\n'
        for i,comp in enumerate(self.mix):
            string+=f'{comp.name} with x= {self.mol_frac[i]:.4f}\n'
        return(string)


def initialize_mixture(components,group_MWs,num_components=5,family_restriction=True):
    """
    Function for initializing the mixture class and randomly selecting possible
    components. Can require a component be from the main chemical families.

    Parameters
    ----------
    components : list of pure component classes
        list of the pure components.
    num_components : int, optional
        Number of components to make the mixture. The default is 5.
    family_restriction : boolean, optional
        Sets the requirement to select from the chemical families.
        The remaining components are randomly selected. The default is True.

    Returns
    -------
    mix a custom mixture class

    """

    mix=mixture(group_MWs)

    if not family_restriction: # Random selection...
        for element in np.random.choice(components,size=num_components,replace=False):
            mix.add_component(1/num_components, element)
        return(mix)

    else: # Split components into familes...
        iso = set()
        paraffin = set()
        cyclic = set()
        aromatic = set()
        olefin = set()
        acetylenic = set()

        for element in components:
            fam = element.mol_type
            if fam=='ISO': iso.add(element)
            elif fam=='PARAFFIN': paraffin.add(element)
            elif fam=='CYCLIC': cyclic.add(element)
            elif fam=='AROMATIC': aromatic.add(element)
            elif fam=='OLEFIN': olefin.add(element)
            elif fam=='ACETYLENIC': acetylenic.add(element)
        if num_components>=1: # select an n-paraffin...
            selection = random.choice(list(paraffin))
            mix.add_component(1/num_components,selection)
            paraffin.remove(selection)
        if num_components>=2: # select an i-paraffin...
            selection = random.choice(list(iso))
            mix.add_component(1/num_components,selection)
            iso.remove(selection)
        if num_components>=3: # select an aromatic...
            selection = random.choice(list(aromatic))
            mix.add_component(1/num_components,selection)
            aromatic.remove(selection)
        if num_components>=4: # select a cyclic...
            selection = random.choice(list(cyclic))
            mix.add_component(1/num_components,selection)
            cyclic.remove(selection)
        # join the remainder and select from this:
        remainder=set().union(iso).union(paraffin).union(cyclic).union(aromatic).union(olefin).union(acetylenic)

        for element in np.random.choice(list(remainder),size=num_components-4,replace=False):
            mix.add_component(1/num_components, element)

        return(mix)

def find_mixtures(target,components,group_MWs,num_components=5,family_restriction=True,DCN_constr=False,MW_constr=False,PC_data=[],threshold=0.01,max_iter=100,max_mixtures=100,user_name='',save_location=''):
    """
    Function for finding mixtures using SLSQP which match target functional group representation.
    Can optionally match other constraints.

    Parameters
    ----------
    target : class fuel_target
        A class containing the functional group and other property data bounds to target
    components : list of pure_components
        list of possible pure components to make the surrogate mixtures
    num_components : int, optional
        The number of components to make the surrogate mixtures. The default is 5.
    family_restriction : boolean, optional
        Set the restriction to fill chemical families. The default is True.
    DCN_constr : boolean, optional
        Set the constraint to match DCN bounds. The default is False.
    MW_constr : boolean, optional
        Set the constraint to match MW bounds. The default is False.
    max_iter : int, optional
        Maximum number of mixture searches. The default is 100.
    max_mixtures : int, optional
        Maximum number of mixtures desired. The default is 100.
    user_name : string, optional
        User name (label for db). The default is ''.
    save_location : pathlib path, optional
        Path to folder to save mixture and fragmenter data. The default is ''.

    Returns
    -------
    mix: a mixture class of the last mixture found

    """

    iterations = 0
    found_mixtures=0
    prev_mixs = []
    prev_mixs_hash = []
    penalty_coefficient = 100

    while iterations<max_iter and found_mixtures<max_mixtures:
        iterations+=1
        mix = initialize_mixture(components,group_MWs,num_components,family_restriction)

        # Constraint function for wt_fraction...
        def fg_min_constr(x,*args):
            mix.update_mol_frac(x)
            return(mix.fragment()-target.fg_mins)

        def fg_max_constr(x,*args):
            mix.update_mol_frac(x)
            return(target.fg_maxs-mix.fragment())

        # Constraint function for DCN...
        def DCN_max_constr(x,*args):
            mix.update_mol_frac(x)
            return(target.DCN[1]-mix.calc_DCN())

        def DCN_min_constr(x,*args):
            mix.update_mol_frac(x)
            return(mix.calc_DCN()-target.DCN[0])

        # Constraint function for MW...
        def MW_min_constr(x,*args):
            mix.update_mol_frac(x)
            return(mix.calc_MW()-target.MW[0])

        def MW_max_constr(x,*args):
            mix.update_mol_frac(x)
            return((target.MW[1])-mix.calc_MW())

        # def MW_constr_func(x,*args):
        #     mix.update_mol_frac(x)
        #     return((target.MW[1])-mix.calc_MW())

        # Define the constraint function...
        def constraint_function(x):
            return np.sum(x) - 1

        # Define the objective function
        objective = lambda x: abs(np.sum(x) - 1) + penalty_coefficient*max(0, constraint_function(x))**2

        cons = []
        #Always add functional group constraints...
        cons.append({'type' : 'ineq' , 'fun': fg_min_constr})
        cons.append({'type' : 'ineq' , 'fun': fg_max_constr})

        if DCN_constr == True:
            cons.append({'type' : 'ineq' , 'fun': DCN_max_constr})
            cons.append({'type' : 'ineq' , 'fun': DCN_min_constr})

        if MW_constr == True:
            cons.append({'type' : 'ineq' , 'fun': MW_min_constr})
            cons.append({'type' : 'ineq' , 'fun': MW_max_constr})
            # cons.append({'type' : 'ineq' , 'fun': MW_constr_func})

        guess = np.ones(num_components)/(num_components+1) # Guess values for mole fractions...
        bounds = tuple([(0, 1) for d in range(num_components)]) # Each weight fraction should be between 0 and 1 for each component...
        result = minimize(objective, x0 = guess, method = 'SLSQP', bounds = bounds, constraints = cons) # Returns the optimized weight fractions...

        if result.success != True or hash(mix) in prev_mixs_hash:
            # print('\nMixtures found: ', found_mixtures)
            # print(f'Iteration: {iterations} failed.\n')
            continue
        elif np.sum(result.x) != 0:
            prev_mixs_hash.append(hash(mix))

            def parse_mixture_string(mix_str):
                parts = mix_str.split(',')
                return {comp.split(':')[0]: float(comp.split(':')[1]) for comp in parts}

            def is_similar_mixture_str(mix_str1, mix_str2, tol=0.8):
                mix1 = parse_mixture_string(mix_str1)
                mix2 = parse_mixture_string(mix_str2)

                if set(mix1.keys()) != set(mix2.keys()):
                    return False

                for comp in mix1:
                    if abs(mix1[comp] - mix2[comp]) > tol:
                        return False

                return True

            def is_string_present(new_mix, mixture_list, tol=0.8):
                for existing_mix in mixture_list:
                    if is_similar_mixture_str(new_mix, existing_mix, tol):
                        return True  # Do not add...
                return False

            current_pred_mix = ''
            for m, n in enumerate(mix.calc_Y()):
                if current_pred_mix!='':
                    current_pred_mix = current_pred_mix+','+str(mix.mix[m]).split(' ')[1]+':'+str(n)
                else:
                    current_pred_mix = current_pred_mix+str(mix.mix[m]).split(' ')[1]+':'+str(n)

            is_present = is_string_present(current_pred_mix, prev_mixs, 0.1) # Adjust for unique mixtures...
            if is_present == False:
                prev_mixs.append(current_pred_mix)
                below_min_indices = np.array([num for num, d in enumerate(mix.calc_Y()) if d<threshold]) # Adjust for trace component mass fractions...
                if len(below_min_indices) == 0:
                    found_mixtures+=1
                post_processing(mix, target, result, found_mixtures, threshold, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location, num_components, user_name)

    return(mix)

def post_processing(mix, target, result, mixtures_found, threshold, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location, num_components, user_name=''):
    """
    This function writes the mixtures and their fragmentation to disc

    Parameters
    ----------
    mix : mixture class
        A mixture to write to disc
    target : fuel_target class
        A fuel target class
    save_location : pathlib path
        Folder for save location.
    user_name : string, optional
        user name. The default is ''.

    Returns
    -------
    None.

    """

    below_min_indices = np.array([num for num, d in enumerate(result.x) if float(d)<threshold])

    if len(below_min_indices)==0:
        total_num_components = len(result.x)
        d = date.today()
        d = d.strftime("%m-%d-%Y")
        surrogate_file = Path(save_location).joinpath(f'surrogate_mixtures_{target.name}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{total_num_components}_MW_{MW_constr}_{num_components}.csv')
        fragment_file = Path(save_location).joinpath(f'surrogate_fragmentation_{target.name}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{total_num_components}_MW_{MW_constr}_{num_components}.csv')

        appending_fragment = fragment_file.exists()
        appending_surrogate = surrogate_file.exists()

        calculated_MW = np.round(mix.calc_MW(),4)
        if DCN_constr:
            calculated_DCN = np.round(mix.calc_DCN(),4)
        mix_frag = np.round(mix.fragment(),8)

        # Creating mixture id...
        if not appending_fragment:
            mixture_id = 'E'+str(total_num_components)+str(10000000+1)[1:]
        else:
            with open(fragment_file,'r',newline='') as f3:
                data = f3.readlines()
                if len(data) != 1:
                    mixture_id = 'E'+str(total_num_components)+str(10000000+int(data[-1].split(',')[0][2:])+1)[1:]
                else:
                    mixture_id = 'E'+str(total_num_components)+str(10000000+1)[1:]

        # Surrogate fragmentations...
        header1=','.join(['mix_id,Fuel_target']+[str(i)+'_weight' for i in PC_data.columns[0:]]+['Date,User_id'])
        with open(fragment_file,'a+',newline='') as f1:
            if not appending_fragment:
                f1.write(header1)

            str2write = f'\n{mixture_id},{target.name},'
            for fg_num in range(len(mix_frag)):
                str2write+=f'{mix_frag[fg_num]},'

            str2write+=f'{calculated_MW},'
            if DCN_constr:
                str2write+=f'{calculated_DCN},'
            else: pass
            str2write+=f'{d},'
            str2write+=f'{user_name}'

            f1.write(str2write)

        # Surrogate mixtures...
        header2= 'mix_id,Fuel_target,Component,mole_fraction,weight_fraction,Date,User_id'
        with open(surrogate_file,'a+',newline='') as f2:
            if not appending_surrogate:
                f2.write(header2)
            for c,comp in enumerate(mix.mix):
                f2.write(f'\n{mixture_id},{target.name},{comp.name},{mix.mol_frac[c]:.4f},{mix.calc_Y()[c]:.4f},{d},{user_name}')

        # print(f'{mix} succeeded. So far {mixtures_found} have been found.\n')
        # print(f'Total: {np.sum(result.x)}\n')

    else:
        def check_within_range(array_2_check, list_min, list_max):
            status=True
            for i in range(len(array_2_check)):
                if array_2_check[i] >= list_min[i] and array_2_check[i] <= list_max[i]:
                    pass
                else:
                    status = False
            return status

        mix.mix = [c for num, c in enumerate(mix.mix) if num not in below_min_indices]
        mix.mol_frac = [c for num, c in enumerate(mix.mol_frac) if num not in below_min_indices]
        mix_frag = np.round(mix.fragment(),8)

        if np.sum(mix.calc_Y())>=0.9999 and np.sum(mix.mol_frac)>=0.9999 and check_within_range(mix_frag, target.fg_mins, target.fg_maxs) == True:
            total_num_components = len(mix.mol_frac)
            d = date.today()
            d = d.strftime("%m-%d-%Y")
            surrogate_file = Path(save_location).joinpath(f'surrogate_mixtures_{target.name}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{total_num_components}_MW_{MW_constr}_{num_components}.csv')
            fragment_file = Path(save_location).joinpath(f'surrogate_fragmentation_{target.name}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{total_num_components}_MW_{MW_constr}_{num_components}.csv')

            appending_fragment = fragment_file.exists()
            appending_surrogate = surrogate_file.exists()

            calculated_MW = np.round(mix.calc_MW(),4)
            if DCN_constr:
                calculated_DCN = np.round(mix.calc_DCN(),4)
            mix_frag = np.round(mix.fragment(),8)

            # # Creating mixture id...
            if not appending_fragment:
                mixture_id = 'E'+str(total_num_components)+str(10000000+1)[1:]
            else:
                with open(fragment_file,'r',newline='') as f3:
                    data = f3.readlines()
                    if len(data) != 1:
                        mixture_id = 'E'+str(total_num_components)+str(10000000+int(data[-1].split(',')[0][2:])+1)[1:]
                    else:
                        mixture_id = 'E'+str(total_num_components)+str(10000000+1)[1:]

            # Surrogate fragmentations...
            header1=','.join(['mix_id,Fuel_target']+[str(i)+'_weight' for i in PC_data.columns[0:]]+['Date,User_id'])
            with open(fragment_file,'a+',newline='') as f1:
                if not appending_fragment:
                    f1.write(header1)

                str2write = f'\n{mixture_id},{target.name},'
                for fg_num in range(len(mix_frag)):
                    str2write+=f'{mix_frag[fg_num]},'

                str2write+=f'{calculated_MW},'
                if DCN_constr:
                    str2write+=f'{calculated_DCN},'
                else: pass
                str2write+=f'{d},'
                str2write+=f'{user_name}'

                f1.write(str2write)

            # Surrogate mixtures...
            header2= 'mix_id,Fuel_target,Component,mole_fraction,weight_fraction,Date,User_id'
            with open(surrogate_file,'a+',newline='') as f2:
                if not appending_surrogate:
                    f2.write(header2)
                for c,comp in enumerate(mix.mix):
                    f2.write(f'\n{mixture_id},{target.name},{comp.name},{mix.mol_frac[c]:.4f},{mix.calc_Y()[c]:.4f},{d},{user_name}')

def create_files(target, num_components, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location=''):
    """
    Function to create dummy files (surrogate mixture and fragmentation) for each n-component mixture...

    Parameters
    ----------
    target : class
        A fuel target class.
    num_components : int
        The number of components in a mixture.
    PC_data : dataframe
        A dataframe of the pure components and its functional group compositions and other details.
    family_restriction : boolean
        True or False.
    DCN_constr : boolean
        True or False.
    MW_constr : boolean
        True or False.
    max_iter : int
        Total number of iterations for each n-component mixture.
    save_location : str, optional
        Save location. The default is ''.

    Returns
    -------
    None.

    """

    for num_comp in num_components:
        surrogate_file = Path(save_location).joinpath(f'surrogate_mixtures_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}.csv')
        fragment_file = Path(save_location).joinpath(f'surrogate_fragmentation_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}.csv')

        appending_surrogate = surrogate_file.exists()
        appending_fragment = fragment_file.exists()

        # Surrogate fragmentations...
        header1=','.join(['mix_id,Fuel_target']+[str(i)+'_weight' for i in PC_data.columns[0:]]+['Date,User_id'])
        with open(fragment_file,'a+', newline='') as f1:
            if not appending_fragment:
                f1.write(header1)

        # Surrogate mixtures...
        header2= 'mix_id,Fuel_target,Component,mole_fraction,weight_fraction,Date,User_id'
        with open(surrogate_file,'a+', newline='') as f2:
            if not appending_surrogate:
                f2.write(header2)

def generator(filepath_to_fuel_properties, PC_data, target, num_components=4, threshold=0.01, save_location='', DCN_constr=False, MW_constr=False, family_restriction=False, max_iter=5000, max_mixtures=10, user_name='N/A'):
    """
    Function to initiate the mixture generation for each n-component mixture...

    Parameters
    ----------
    filepath_to_fuel_properties : str
        Path location for fuel functional group compositions.
    PC_data : dataframe
        A pandas dataframe of the pure components and its functional group compositions and other details.
    target : class
        A fuel target class.
    num_components : int, optional
        The number of components in a mixture. The default is 4.
    threshold : float, optional
        Minimum mole fraction for each component in a mixture. The default is 0.01.
    save_location : str, optional
        Save location. The default is ''.
    DCN_constr : boolean, optional
        True or False. The default is False.
    MW_constr : boolean, optional
        True or False. The default is False.
    family_restriction : boolean, optional
        True or False. The default is False.
    max_iter : int, optional
        Total number of iterations for each n-component mixture. The default is 5000.
    max_mixtures : int, optional
        Total number of mixtures to be found for each n-component mixture. The default is 10.
    user_name : str, optional
        Provide the user name. The default is 'N/A'.

    Returns
    -------
    None.

    """

    global num_FGs
    num_FGs = len(PC_data.columns[:-1])

    group_MWs, components = read_component_db(PC_data, DCN_constr, MW_constr, family_restriction)
    fuels = read_fuels_db(filepath_to_fuel_properties, PC_data, DCN_constr, MW_constr)

    # Check if total number of functional group matches...
    fuel_temp_read = pd.read_csv(filepath_to_fuel_properties, header=0, index_col=0)
    if num_FGs != len(fuel_temp_read.columns[:-2]):
        raise ValueError("Length of functional groups in pure component file and fuel properties file does not match. Please check files containing pure component functional group distribution and file containing functional group distribution of fuel. Make sure to modify the total number of functional groups in fuels file to match to the functional groups in pure components file.")

    # Specify locations to save files...
    save_location = Path(save_location)

    find_mixtures(fuels[0],components,group_MWs,num_components,family_restriction,DCN_constr,MW_constr,PC_data,threshold,max_iter,max_mixtures,user_name,save_location)

def rearrange_files(target, num_components, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location):
    """
    Function to combine all n-component surrogate mixtures to their respective categories...

    Parameters
    ----------
    target : class
        A fuel target class.
    num_components : int
        The number of components in a mixture.
    PC_data : dataframe
        A dataframe of the pure components and its functional group compositions and other details.
    family_restriction : boolean
        True or False.
    DCN_constr : boolean
        True or False.
    MW_constr : boolean
        True or False.
    max_iter : int
        Total number of iterations for each n-component mixture.
    save_location : str, optional
        Save location. The default is ''.

    Returns
    -------
    None.

    """

    for num_comp in tqdm(num_components):
        surrogate_filepath = Path(save_location).joinpath(f'surrogate_mixtures_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}.csv')
        fragment_filepath = Path(save_location).joinpath(f'surrogate_fragmentation_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}.csv')

        # Check if paths exist...
        surrogate_exist = surrogate_filepath.exists()
        fragment_exist = fragment_filepath.exists()

        if not fragment_exist or not surrogate_exist:
            create_files(target, [num_comp], PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location)

        surrogate_df = pd.read_csv(surrogate_filepath, header=0, index_col=None)
        fragment_df = pd.read_csv(fragment_filepath, header=0, index_col=None)

        for nc in num_components:
            if nc>=num_comp:
                surrogate_filepath_temp = Path(save_location).joinpath(f'surrogate_mixtures_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}_{nc}.csv')
                fragment_filepath_temp = Path(save_location).joinpath(f'surrogate_fragmentation_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}_{nc}.csv')

                # Check if paths exist...
                appending_surrogate_exist = surrogate_filepath_temp.exists()
                appending_fragment_exist = fragment_filepath_temp.exists()

                if appending_fragment_exist and appending_surrogate_exist:
                    # Skipping inconsistent columns...
                    surrogate_file = pd.read_csv(surrogate_filepath_temp, skiprows = 0, header=None, on_bad_lines='skip')
                    no_cols = len(surrogate_file.columns)
                    # Reading file again without the inconsistent column...
                    surrogate_file = pd.read_csv(surrogate_filepath_temp, skiprows = 0, header=0, usecols=range(no_cols))

                    # Skipping inconsistent columns...
                    fragment_file = pd.read_csv(fragment_filepath_temp, skiprows = 0, header=None, on_bad_lines='skip')
                    no_cols = len(fragment_file.columns)
                    # Reading file again without the inconsistent column...
                    fragment_file = pd.read_csv(fragment_filepath_temp, skiprows = 0, header=0, usecols=range(no_cols))

                    # Read and combine...
                    # Surrogate mixtures...
                    df_surrogate_list = [surrogate_df, surrogate_file]  # Read each CSV into a DataFrame...
                    surrogate_df = pd.concat(df_surrogate_list, ignore_index=True)  # Concatenate them vertically...
                    # Surrogate mixture fragmentations...
                    df_frag_list = [fragment_df, fragment_file]  # Read each CSV into a DataFrame...
                    fragment_df = pd.concat(df_frag_list, ignore_index=True)  # Concatenate them vertically...

        # Save the combined DataFrame into a new CSV file...
        surrogate_df.to_csv(surrogate_filepath, index=False)
        fragment_df.to_csv(fragment_filepath, index=False)

# Remove temparory files...
def remove_temp_files(target, num_components, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location):
    """
    Function to remove the temporary files...

    Parameters
    ----------
    target : class
        A fuel target class.
    num_components : int
        The number of components in a mixture.
    PC_data : dataframe
        A dataframe of the pure components and its functional group compositions and other details.
    family_restriction : boolean
        True or False.
    DCN_constr : boolean
        True or False.
    MW_constr : boolean
        True or False.
    max_iter : int
        Total number of iterations for each n-component mixture.
    save_location : str, optional
        Save location. The default is ''.

    Returns
    -------
    None.

    """

    for num_comp in tqdm(num_components):
        for nc in num_components:
            if nc>=num_comp:
                surrogate_filepath_temp = Path(save_location).joinpath(f'surrogate_mixtures_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}_{nc}.csv')
                fragment_filepath_temp = Path(save_location).joinpath(f'surrogate_fragmentation_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}_{nc}.csv')
                # Check if paths exist...
                appending_surrogate_exist = surrogate_filepath_temp.exists()
                appending_fragment_exist = fragment_filepath_temp.exists()

                if appending_fragment_exist and appending_surrogate_exist:
                    os.remove(Path(surrogate_filepath_temp))
                    os.remove(Path(fragment_filepath_temp))

        surrogate_filepath = Path(save_location).joinpath(f'surrogate_mixtures_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}.csv')
        fragment_filepath = Path(save_location).joinpath(f'surrogate_fragmentation_{target}_{max_iter}_iterations_DCN_{DCN_constr}_family_restriction_{family_restriction}_Number_components_{num_comp}_MW_{MW_constr}.csv')

        # Check if paths exist...
        surrogate_exist = surrogate_filepath.exists()
        fragment_exist = fragment_filepath.exists()

        if surrogate_exist and fragment_exist:
            surrogate_df = pd.read_csv(surrogate_filepath, header=0, index_col=None)
            fragment_df = pd.read_csv(fragment_filepath, header=0, index_col=None)

            if len(surrogate_df) == 0 and len(fragment_df) == 0:
                os.remove(Path(surrogate_filepath))
                os.remove(Path(fragment_filepath))

def main(num_components: int, filepath_to_fuel_properties='', PC_data=None, fuel_name='', threshold=0.01, save_location='', DCN_constr=False, MW_constr=False, family_restriction=False, max_iter=5000, max_mixtures=10, user_name=''):
    """
    Function for multiprocessed generation of all n-component mixtures...

    Parameters
    ----------
    num_components : list of integers
        The number of components in a mixture. The default is [4].
    filepath_to_fuel_properties : str, optional
        Path location for fuel functional group compositions. The default is ''.
    PC_data : dataframe, optional
        A pandas dataframe of the pure components and its functional group compositions and other details. The default is None.
    fuel_name : str, optional
        Name of the fuel for which the surrogates are made. The default is ''.
    threshold : float, optional
        Minimum mole fraction for each component in a mixture. The default is 0.01.
    save_location : str, optional
        Save location. The default is ''.
    DCN_constr : boolean, optional
        True or False. The default is False.
    MW_constr : boolean, optional
        True or False. The default is False.
    family_restriction : boolean, optional
        True or False. The default is False.
    max_iter : int, optional
        Total number of iterations for each n-component mixture. The default is 5000.
    max_mixtures : int, optional
        Total number of mixtures to be found for each n-component mixture. The default is 10.
    user_name : str, optional
        Provide the user name. The default is ''.

    Returns
    -------
    None.

    """

    # Create empty files...
    create_files(fuel_name, num_components, PC_data, family_restriction, DCN_constr, MW_constr, max_iter, save_location)

    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(generator, filepath_to_fuel_properties, PC_data, fuel_name, num_comp, threshold, save_location, DCN_constr, MW_constr, family_restriction, max_iter, max_mixtures, user_name) for num_comp in num_components]
        for future in tqdm(as_completed(futures), total=len(num_components)):
            future.result()