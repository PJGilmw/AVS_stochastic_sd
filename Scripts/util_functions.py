# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:54:48 2024

@author: pierre.jouannais

Script containing accessory functions needed by other functions across the different scripts.
"""

import bw2data as bd
import bw2calc as bc
import bw2analyzer as ba
import bw2io as bi
import numpy as np
from scipy import sparse

import bw_processing as bwp

import matrix_utils as mu
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

import pandas as pd


import pickle


import math 
from ema_workbench.analysis import prim

from ema_workbench import ema_logging

import datetime
import os

import json


ema_logging.log_to_stderr(ema_logging.INFO)

def is_defined(var_name):
    """Checks if a variable is defined"""
    
    global_vars = globals()
    return var_name in global_vars

def export_pickle_2(var, name_var, namefolder_in_root):
    '''Saves a pickle in the working directory and
    saves the object in input across python sessions'''

    path_object = "../"+namefolder_in_root+"/"+name_var+".pkl"
    with open(path_object, 'wb') as pickle_file:
        pickle.dump(var, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)



def importpickle(path):
    """Imports a pickle object"""

    with open(path, 'rb') as pickle_load:
        obj = pickle.load(pickle_load)
    return obj  


def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it doesn't exist.
    
    Args:
    - folder_path: The path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")




def write_json_file(file_path, data):
    """
    Write data to a JSON file.
    
    Args:
    - file_path: The path of the JSON file to write.
    - data: The data to write to the JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data written to '{file_path}' successfully.")







def fix_switch_valuedatapackages(values_for_datapackages): 
    
    """Removes the switch parameters from the dictionnary of values"""
    
    switch_to_del=[]
    for a in values_for_datapackages:
        if "switch" in a:
            switch_to_del.append(a)
            
    for a in switch_to_del:
        values_for_datapackages.pop(a)
        
    return values_for_datapackages    


def load_characterization_matrix(list_meth,Lca_object):
    
    """Collects the characterization matrixes in a dictionnary"""

    
    C_matrixes={}
    for meth_ in list_meth:
        Lca_object.switch_method(meth_)

        C_matrixes[meth_]=Lca_object.characterization_matrix
        
    return C_matrixes



# CHECK IF NEEDED
def name_modif_meth(name_meth):
    
    
    name_meth0=name_meth[0]
    name_meth1=name_meth[1]
    name_meth2=name_meth[2]
    
    new_name=(name_meth[0],name_meth[1],name_meth[2]+"m")
    
    return new_name


def add_virtual_flow_to_additional_biosphere(list_meth):
    
    """For a list of impact categories, creates an artificial biosphere 
    containing flows fow which 1 unit has a characterization factor of 1 for the impact category """
    
    if is_defined("additional_biosphere_multi_categories"):
        print("Delete former db")
        del bd.databases["additional_biosphere_multi_categories"]
        
    dict_virtual_act = {}
    
    for meth in list_meth:
        print(meth)

        meth_code = meth[-1]
        dict_virtual_act[('additional_biosphere_multi_categories', meth_code)]={
                                                                                'name': meth_code,
                                                                                'unit': 'virtual impact category unit',  # Define the unit for your substance
                                                                                'type': 'emission'}

        
    additional_biosphere_multi_categories = bd.Database('additional_biosphere_multi_categories')
    additional_biosphere_multi_categories.write(dict_virtual_act)
    
    list_modif_meth=[]
    
    list_virtual_emissions_same_order_as_meth = []
    print("iii")
    for meth in list_meth:
        
        meth_code = meth[-1]
        print(meth)

        meth_data = bd.Method(meth).load() 
        
        modif_meth_data = meth_data+[(additional_biosphere_multi_categories.get(meth_code).id, 1.0)]

        meth_modif_key = name_modif_meth(meth)

        modifmeth = bd.Method(meth_modif_key)
        modifmeth.register()
        modifmeth.write(modif_meth_data)
        modifmeth.load()
        
        list_modif_meth.append(meth_modif_key)
        
        # get the act
        
        act_meth=additional_biosphere_multi_categories.get(meth_code)
        list_virtual_emissions_same_order_as_meth.append(act_meth)
        
        
        
    return additional_biosphere_multi_categories,list_modif_meth,list_virtual_emissions_same_order_as_meth


def init_act_variation_impact(act_copy,list_meth,additional_biosphere_multi_categories):
    
    """For a list of impact categories, modifies an activity "act_copy"" by adding one output flow for each impact category, with a CF of 1"""

        
    lca1 = bc.LCA({act_copy: 1}, list_meth[1]) # as it works on list_meth and not list_meth_modif, there is no double accounting
    lca1.lci()
    lca1.lcia()
    print("xxxx")
    list_exchanges = [exc["input"][1] for exc in list(act_copy.exchanges())]
    print(list_exchanges)
    for meth_ in list_meth:
        
        meth_code = meth_[-1]
        
        if meth_code not in list_exchanges: #avoid adding exchanges that were already added
            print("meth_code",meth_code)
            lca1.switch_method(meth_)
            lca1.lci()
            lca1.lcia()
            
            print(lca1.score)
            act_input = additional_biosphere_multi_categories.get(meth_code)
     
            act_copy.new_exchange(amount=lca1.score, input=act_input, type="biosphere").save()
        
    act_copy.save()
    
    

    return act_copy




def extract_prefix(s):
    if "_switch" in s:
        return s.split("_switch")[0]
    else:
        return s

def find_crop(row,colnames_switch):
    
    """ For one row of the sample dataframe, returns the crop ( integer) that was simulated out of the 3 switch parameters"""

    
    #print(row)
    corres_int_string = [(index,a) for index,a in enumerate(colnames_switch)]

    value_cropswitch=None
    for colname in corres_int_string:
        #print(row[colname])

        value =  row[colname[1]]
        if value==1:
            value_cropswitch=colname[0]
            
    
    return value_cropswitch, corres_int_string  



def melt_switch(sample_dataframe,colnames_switch,name_meltcolumn):
    
    """ Replaces the original columns for the switch parameters by a unique column giving the value of the categorical variable (integer)"""

    
    sample_dataframe_melt = sample_dataframe.copy()
    sample_dataframe_melt[name_meltcolumn]= [find_crop(sample_dataframe.iloc[i],colnames_switch)[0] for i in range(len(sample_dataframe))]
    sample_dataframe_melt.drop(colnames_switch, axis=1, inplace=True)

    return sample_dataframe_melt


def find_crop_names(row,colnames_switch):
    
    """ For one row of the sample dataframe, returns the crop (name) that was simulated out of the 3 switch parameters"""
    #print(row)
    corres_int_string = [(index,a) for index,a in enumerate(colnames_switch)]

    value_cropswitch=None
    for colname in corres_int_string:
        #print(row[colname])

        value =  row[colname[1]]
        if value==1:
            value_cropswitch=colname[1]
            
    
    return value_cropswitch, corres_int_string  



def melt_switch_names(sample_dataframe,colnames_switch,name_meltcolumn):

    """ Replaces the original columns for the switch parameters by a unique column giving the value of the categorical variable (name)"""

    sample_dataframe_melt = sample_dataframe.copy()
    sample_dataframe_melt[name_meltcolumn]= [find_crop_names(sample_dataframe.iloc[i],colnames_switch)[0] for i in range(len(sample_dataframe))]
    sample_dataframe_melt.drop(colnames_switch, axis=1, inplace=True)

    return sample_dataframe_melt




# REDUCE SIZE



def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float16)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist







def reduce_mem_usage_only16(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            

                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            

            
            # Make float datatypes 16 bit
            if not IsInt:
                props[col] = props[col].astype(np.float16)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

