# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:54:48 2024

@author: pierre.jouannais


Script that computes the stochastic LCAs.
Saves intermediates stochastic results in results_intermediate. They can be imported for the next iterations of prim.


The whole script must be excecuted after changing the settings if necessary (l 20) 
and the path for the intermediate results if necessary (l 175)


"""



############
""" Define the settings for the Monte Carlo iterations and PRIM.
The current values are the ones used"""
############




# Number of MC iterations (if not imported)
size=400000
        



# To include background uncertainty. NOT READY YET.
uncert=False




# Choose impact categories

meth1 = ('ReCiPe 2016 v1.03, midpoint WITH ILUC (H)',
              'climate change',
              'global warming potential (GWP100) ILUC')

meth2 =('ReCiPe 2016 v1.03, midpoint (H)',
 'material resources: metals/minerals',
 'surplus ore potential (SOP)')

meth3= ('ReCiPe 2016 v1.03, midpoint (H)',
  'water use',
  'water consumption potential (WCP)')


meth4=('ReCiPe 2016 v1.03, midpoint (H)',
  'particulate matter formation',
  'particulate matter formation potential (PMFP)')

meth5= ('ReCiPe 2016 v1.03, midpoint (H)',
  'eutrophication: freshwater',
  'freshwater eutrophication potential (FEP)')

meth6= ('ReCiPe 2016 v1.03, midpoint (H)',
  'land use',
  'agricultural land occupation (LOP)')

meth7= ('ReCiPe 2016 v1.03, midpoint (H)',
  'ecotoxicity: terrestrial',
  'terrestrial ecotoxicity potential (TETP)')


list_meth = [meth1,meth2,meth3,meth4,meth5,meth6,meth7]

# When looking for a total success/failure, which impact categories are necessary?

index_meth_conditions=[0,1,2,3,4,5,6] # all indexes = all impact categories

##################################
##################################




import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)
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

from Main_functions import *
import Parameters_and_functions
from Parameters_and_functions  import *

from Plot_functions import *
import util_functions

import pickle


import math 
from ema_workbench.analysis import prim


import datetime
import os

import json

import ray

import time

        

ray.shutdown()

ray.init()

num_cpus = int(ray.available_resources()["CPU"])












 

""" Compute Stochastic LCAs"""


""" Load project, db, methods etc."""

# Check if needed or if it's imported with the other scripts
bd.projects.set_current('AVS')


Ecoinvent = bd.Database('ecoinvent-3.10-consequential')

biosphere = bd.Database('ecoinvent-3.10-biosphere')


foregroundAVS = bd.Database("foregroundAVS")


# Create a parameterized marginal elec according to the choice of meths
additional_biosphere_multi_categories,list_modif_meth,list_virtual_emissions_same_order_as_meth =  add_virtual_flow_to_additional_biosphere(list_meth)
elec_marginal_fr_copy = init_act_variation_impact(elec_marginal_fr_copy,list_meth,additional_biosphere_multi_categories)

dict_funct["modif_impact_marginal_elec"]={"func":vectorize_function(modif_impact_marginal_elec),
                     "indices":[(act_meth.id,elec_marginal_fr_copy.id) for act_meth in list_virtual_emissions_same_order_as_meth]}



# Collect all indices/positions in the LCA matrixes on which functions will apply

list_totalindices = []                                 
for a in dict_funct:
    
    list_totalindices= list_totalindices+dict_funct[a]["indices"]



# MC iterations for the parameters. 

names_param_total, sample_dataframe= sampling_func_lhs(dictionnaries,
                             size)

# Put the input sample in a dictionnary with all the sampled values for the input parameters

values_for_datapackages = {}

for dict in dictionnaries:
    
   values_for_datapackages = Merge(values_for_datapackages , dictionnaries[dict])
    
   
values_for_datapackages= fix_switch_valuedatapackages(values_for_datapackages)

for param in sample_dataframe:
    
    values_for_datapackages[param]={"values":sample_dataframe[param].tolist()}




"""Create the modified datapackage"""


# The datapackage will contain the LCA matrixes 
# with the list of stochastic values calculated according to the parameters values and functions.

list_arrays_for_datapackages,values_for_datapackages,list_chunk_sizes = create_modif_arrays_para(AVS_elec_main,
                          meth1,
                          "foregroundAVS",
                          'ecoinvent-3.10-consequential',
                          "additional_biosphere",
                          "additional_biosphere_multi_categories",
                          list_totalindices,
                          dict_funct,
                          values_for_datapackages,
                          num_cpus)




"""Compute the LCAs"""

# List of FUs

list_fu=[[AVS_elec_main.id,"AVS_elec_main"],
         [PV_ref.id,"PV_ref"],
         [AVS_crop_main.id,"AVS_crop_main"],
         [wheat_fr_ref.id,"wheat_fr_ref"],
         [soy_ch_ref.id,"soy_ch_ref"],
         [alfalfa_ch_ref.id,"alfalfa_ch_ref"],
         [elec_marginal_fr_copy.id,"elec_marginal_fr_copy"]] 



list_tables = compute_stochastic_lcas_para(
                            list_arrays_for_datapackages,
                            list_fu,
                            AVS_elec_main,
                            list_modif_meth,
                            uncert,
                            list_chunk_sizes,
                            indices_array_fix,
                            data_array_fix)



""" Export intermediate results"""

x = datetime.datetime.now()

month=str(x.month)
day=str(x.day)
microsec=str(x.strftime("%f"))
             


name_file_res ='listtablesresultpara'+"_"+month+"_"+day+"_"+microsec+"size="+str(size)
name_file_param ='listtablesparampara'+"_"+month+"_"+day+"_"+microsec+"size="+str(size)
name_file_meth='listmethpara'+"_"+month+"_"+day+"_"+microsec+"size="+str(size)


export_pickle_2(list_tables, name_file_res, "resultsintermediate")
export_pickle_2(sample_dataframe, name_file_param, "resultsintermediate")
export_pickle_2(list_meth, name_file_meth, "resultsintermediate")





