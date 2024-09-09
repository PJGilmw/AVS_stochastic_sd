# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:54:48 2024

@author: pierre.jouannais


Script that applies PRIM over the stochastic sample that is firt imported.

Saves the results in the folder PRIM_process. 

The whole script must be excecuted after changing the settings if necessary (l 20) 
and the path for the intermediate results if necessary (l 175)


"""



############
""" Define the settings for the Monte Carlo iterations and PRIM.
The current values are the ones used"""
############










#min_mass_points=2 # Minimum number of data points for a box in prim

type_delta="percent_modif" # Type of comparison: percentage of modification (A X% better or worse than B). Do not change. 

act1="AVS_elec_main" # Comparison tested = act 1 - act 2. Do not change
act2="PV_ref" # Comparison tested = act 1 - act 2. Do not change


# List of prim parameters to test. 
# The function applies prim in series with different settings.

list_densities=[0.8] # Minimal density of a box for prim
list_mode_compa=["inf","inf","sup"] # Type of comparison. AVS inferior to Conv or infer 

# Delta for comparison. in fraction of impact. 
#Here For instancs AVS < (1-1/3) * Conv, AVS < Conv, AVS > (1+1/3) * Conv
list_delta=[1/3,0,1/3] 

list_mode_prim =["default"] # default or guivarch. Check prim documentation.
list_alpha = [0.05] # Share of data points peeled for each peeling step of prim

# Plot intermediate outputs of prim (peeling trajectories etc.)
# Only set it to True for debugging, otherwise, too heavy.
plot= False 

# To include background uncertainty. NOT READY YET.
uncert=False


# Maximum number of boxes to be found by prim
max_boxes=200



    
# Minimum percentage of  data points per box    
min_mass = 0.0002 


# Just to indicate  in the saved export if we assumed a fixed PV technology set at the current baseline
fixedornot ="var" # or "fixed"


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

index_meth_conditions=[0,1,2,3,4,5,6] # all impact categories

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

#from ema_workbench import ema_logging

import datetime
import os

import json

import ray

import time
#ema_logging.log_to_stderr(ema_logging.INFO)

        

ray.shutdown()

ray.init()

#num_cpus = int(ray.available_resources()["CPU"])

num_cpus=5 # To adapt depending on the available memory. 





#ema_logging.log_to_stderr(ema_logging.INFO)



""" Load previously exported stochastic sample in resultsintermediate"""

    
list_meth= importpickle("../resultsintermediate/listmethpara400k_7_23_848065size=400000.pkl")

sample_dataframe= importpickle("../resultsintermediate/listtablesparam400k_7_23_848065size=400000.pkl")

list_tables= importpickle("../resultsintermediate/listtablesresult400k_7_23_848065size=400000.pkl")




    

    

    
# Here we convert the 3 columns "wheat_switch", "soy_switch", "alfalfa_switch" with 0 or 1.
# into a column "crop" with the name of the crop
colnames_switch =  ["wheat_switch","soy_switch","alfalfa_switch"] 

name_meltcolumn="crop_switch"   # Needs "switch" in the name

sample_dataframe_melt =  melt_switch_names(sample_dataframe,colnames_switch,name_meltcolumn)



# Reduce to 16 bits float to save memory
sample_dataframe_melt,_= reduce_mem_usage_only16(sample_dataframe_melt)



##############################################
##############################################

"""Apply prim"""

##############################################
##############################################




""" Both hypotheses for main product """

time1=time.time()

main="elec" # or "both", or "crop"

dict_results_all_main_elec=apply_prim_different_settings_all_main_parallel(list_alpha,
                                  list_delta,
                                  list_mode_compa,
                                  list_densities,
                                  list_mode_prim,
                                  list_tables,
                                  sample_dataframe_melt,
                                  act1,
                                  act2,
                                  plot,
                                  min_mass,
                                  type_delta,
                                  list_meth,
                                  index_meth_conditions,
                                  main,
                                  max_boxes,
                                  name_meltcolumn,
                                  colnames_switch,
                                  fixedornot)

dict_results_all_main_elec["info"]={"list_alpha":list_alpha,
                               "list_delta":list_delta,
                               "list_mode_compa":list_mode_compa,
                               "list_densities":list_densities,
                               "list_mode_prim":list_mode_prim,
                               "min_mass":min_mass,
                               "type_delta":type_delta,
                               "list_meth":list_meth,
                               "index_meth_conditions":index_meth_conditions,
                               "main":main,
                               "plot":plot,
                               "uncert":uncert,
                               "fixedornot" : fixedornot}

timetot=time.time()-time1

print(timetot)

x = datetime.datetime.now()

month=str(x.month)
day=str(x.day)
microsec=str(x.strftime("%f"))

name_dict_pickle =  "dict_results_all_main_elec_400k"+str(month)+str(day)+str(microsec)+"main"+str(main)                               


export_pickle_2(dict_results_all_main_elec, name_dict_pickle, "PRIM_process/dict_results")




