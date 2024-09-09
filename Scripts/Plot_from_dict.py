# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:19:42 2024

@author: pjouannais

Scripts to excecute to plot the scenarios from the results dictionnaries.
"""



import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt


from Main_functions  import *

from Plot_functions import *

import pickle


import math 
from ema_workbench.analysis import prim

#from ema_workbench import ema_logging

import datetime
import os

import json

import ray


def colnames_switch_parameters(sample_dataframe,colnames_switch):
    
    """ Converts the 3 colunms for crop type into one with the crop type name"""
    
    

    sample_dataframe_melt = sample_dataframe.copy()
    
    sample_dataframe_melt["crop_switch"]= [find_crop(sample_dataframe.iloc[i],colnames_switch)[0] for i in range(len(sample_dataframe))]
    
    sample_dataframe_melt.drop(colnames_switch, axis=1, inplace=True)
    
    return sample_dataframe_melt





   
    
###
"""Import results"""
###


###Intermediate results
list_meth_varpv= importpickle("../resultsintermediate/listmethpara400k_7_23_848065size=400000.pkl")

sample_dataframe_varpv= importpickle("../resultsintermediate/listtablesparam400k_7_23_848065size=400000.pkl")

list_tables_varpv= importpickle("../resultsintermediate/listtablesresult400k_7_23_848065size=400000.pkl")

# Boxes
dictres_import_elec_varpv= importpickle("../PRIM_process/dict_results/400k/dict_results_all_main_elec_400k_lhsvalid.pkl")


# Converts the 3 colunms for crop type into one with the crop type name

colnames_switch =  ["wheat_switch","soy_switch","alfalfa_switch"] 

sample_dataframe_melt_varpv = colnames_switch_parameters(sample_dataframe_varpv,colnames_switch)







typesim = "varpv" # just an indicator in the name of the output file


plot_from_dict_results_fix_switch_categ(dictres_import_elec_varpv,sample_dataframe_melt_varpv,550,1.2,list_meth_varpv,colnames_switch,typesim)
    


