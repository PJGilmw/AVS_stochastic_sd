# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 00:15:42 2024

@author: pjouannais

Script that imports the boxes discovered by PRIM and check them by calculating the actual densities of a selection of them. 
Exports a new dictionnary of boxes after keeping only the ones with a high enough actual density.

The whole script must be excecuted after changing the import path for the original boxes (l 77) 
and the settings results if necessary (l 175)



"""


        

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

from functools import reduce



####
"""Import the original boxes"""
####

dictres_import_elec_varpv= importpickle("../PRIM_process/dict_results/400k/dict_results_all_main_elec_400k728509021mainelec.pkl")


##
"""Double-checking functions"""
##

            
def reset_samplingdict_with_box(dictionnaries_ori,box):
    
    
    dictionnaries = copy.deepcopy(dictionnaries_ori)
    # collect switch categ limits
    list_=["wheat","soy","alfalfa"]
    collect_switch_categ = box["crop_switch"][0]
    
    prob = 1/len(collect_switch_categ)
    
    
    
    in_list=[any([crop in switch for switch in collect_switch_categ]) for crop in list_]
    prob_switch = [i * prob for i in in_list]
    
    for dict_key in dictionnaries.keys():
        
        dict_= dictionnaries[dict_key]
    
        for paramkey in dict_:
            #print(paramkey)
    
            if "switch" not in paramkey:
                #print(paramkey)
    
            
                dict_[paramkey][1][1]= box[paramkey][0]     # lower
                dict_[paramkey][1][2]= box[paramkey][1]     # upper
                
            elif paramkey=="prob_switch_crop":
                #print(dict_[paramkey][2])
                dict_[paramkey][2]= prob_switch    
                
    return dictionnaries         
    
                



            


def check_actual_density_box(box,
                             index_meth,
                             delta,
                             mode_compa,
                             main,
                             size,
                             dictionnaries,
                             meth1,
                             uncert,
                             dict_funct,
                             num_cpus,
                             list_meth,
                             elec_marginal_fr_copy,
                             Ecoinvent,
                             biosphere,
                             foregroundAVS,                  
                             type_delta,
                             act1,
                             act2,
                             list_totalindices
                             
                             ):
        
    
    """ Compute Stochastic LCAs"""
    

    
    
    
    
    
    
    dictionnaries = reset_samplingdict_with_box(dictionnaries,box)     
    
    print("IIIIIIOOOOOO")
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
    
    #return values_for_datapackages, dict_funct, list_totalindices
    
    
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
    
    # Compute stochastic LCAs
    

    
    list_tables = compute_stochastic_lcas_para(
                                list_arrays_for_datapackages,
                                list_fu,
                                AVS_elec_main,
                                list_modif_meth,
                                uncert,
                                list_chunk_sizes,
                                indices_array_fix,
                                data_array_fix)
    
        
    
        
    # Here we convert the 3 columns "wheat_switch", "soy_switch", "alfalfa_switch" with 0 or 1.
    # into a column "crop" with the name of the crop
    colnames_switch =  ["wheat_switch","soy_switch","alfalfa_switch"] 
    
    name_meltcolumn="crop_switch"   # Needs THE "switch" in the name
    
    
    sample_dataframe_melt =  melt_switch_names(sample_dataframe,colnames_switch,name_meltcolumn)
    
    
    #y_elec_main,y_crop_main,

    tables_for_PRIM= prepare_outputs_all_main(list_tables,
                        sample_dataframe_melt,
                        delta,
                        type_delta,
                        mode_compa,
                        act1,
                        act2,
                        name_meltcolumn,
                        colnames_switch) 
    

    
    if main== "both":
        
        table_both_percateg=[table[1]*table[2] for table in tables_for_PRIM]

        if index_meth =="all":
            
            successes = reduce(lambda x, y: x*y, table_both_percateg).sum()
            
        else : 
            
            successes=table_both_percateg[index_meth].sum()
            
    elif main=="crop":
        
        if index_meth =="all":
            
            successes = reduce(lambda x, y: x[2]*y[2], tables_for_PRIM).sum()
            
        else:
            
            successes = tables_for_PRIM[index_meth][2].sum()
            
    elif main=="elec":
        
        if index_meth =="all":
            
            successes = reduce(lambda x, y: x[1]*y[1], tables_for_PRIM).sum()
            
        else:
            
            successes = tables_for_PRIM[index_meth][1].sum() 
            
            
    actual_density = successes/size     


    return actual_density
            







        


def check_boxes_of_dict(dictres_import_ori,
                        main,
                        size,
                        listmeth,
                        sampling_rate_boxes,
                        dictionnaries,
                        meth1,
                        uncert,
                        dict_funct,
                        num_cpus,
                        elec_marginal_fr_copy,
                        Ecoinvent,
                        biosphere,
                        foregroundAVS,
                        type_delta,
                        act1,
                        act2,
                        list_totalindices):
    

     
     dictres_import = copy.deepcopy(dictres_import_ori)
     dictres_import.pop("info")
     
     dict_results = {}
     for key in dictres_import.keys():
         #print(key)
         
         dict_res = dictres_import[key]
         
         list_list_boxes_categories = dict_res[0]
         list_boxes_total = dict_res[1]
        
         
         compa, delta = extract_compa_delta_main(key)

         # Total success
         index_meth = "all"
         
         info_initial_all = list_boxes_total.pop(0)
         
         dict_total_actualdensity = {}
         dict_total_actualdensity["info_init"] = info_initial_all
         
         print(len(list_boxes_total))
         
         correct_samplingrate = min([sampling_rate_boxes, len(list_boxes_total)])        
         
         if correct_samplingrate == len(list_boxes_total):
             
             list_boxnumber = [i for i in range(0,len(list_boxes_total))]
             
         else:
             
             #list_boxnumber = [i for i in range(0,len(list_boxes_total), correct_samplingrate)]
             list_boxnumber =  list(np.unique([i for i in range(0,len(list_boxes_total), correct_samplingrate)]+[len(list_boxes_total)-4,len(list_boxes_total)-3,len(list_boxes_total)-2,len(list_boxes_total)-1]))
          
      
             
             
         for box_number in list_boxnumber:
             
             print(index_meth)
             
             print("boxnumber",box_number)

             box = list_boxes_total[box_number]
             
             dict_total_actualdensity[box_number] = check_actual_density_box(box,
                             index_meth,
                             delta,
                             compa,
                             main,
                             size,
                             dictionnaries,
                             meth1,
                             uncert,
                             dict_funct,
                             num_cpus,
                             listmeth,
                             elec_marginal_fr_copy,
                             Ecoinvent,
                             biosphere,
                             foregroundAVS,
                             type_delta,
                             act1,
                             act2,
                             list_totalindices)
         # Each meth
         list_dict_actualdensitities_meth = []
         
         for index_meth in range(len(listmeth)):
             
             print(index_meth)
             
             
             list_boxes_meth = list_list_boxes_categories[index_meth]
             
             info_initial_meth = list_boxes_meth.pop(0)

             category = listmeth[index_meth][-1]
             
             category_accronym = extract_meth_accronym(category)
             
             dict_actual_density_meth = {}
             
             dict_actual_density_meth["info_init"] = info_initial_meth
             
             correct_samplingrate = min([sampling_rate_boxes, len(list_boxes_meth)])        

             
             if correct_samplingrate == len(list_boxes_meth):
                 
                 list_boxnumber = [i for i in range(0,len(list_boxes_meth))]
                 
             else:
                 
                 #list_boxnumber = [i for i in range(0,len(list_boxes_total), correct_samplingrate)]
                 list_boxnumber =  list(np.unique([i for i in range(0,len(list_boxes_meth), correct_samplingrate)]+[len(list_boxes_meth)-4,len(list_boxes_meth)-3,len(list_boxes_meth)-2,len(list_boxes_meth)-1]))
              
            
                 
                 
             for box_number in list_boxnumber:
                 
                 print("boxnumber",box_number)

                 box = list_boxes_meth[box_number]
                 
                 dict_actual_density_meth[box_number]=check_actual_density_box(box,
                                              index_meth,
                                              delta,
                                              compa,
                                              main,
                                              size,
                                              dictionnaries,
                                              meth1,
                                              dict_funct,
                                              num_cpus,
                                              listmeth,
                                              elec_marginal_fr_copy,
                                              Ecoinvent,
                                              biosphere,
                                              foregroundAVS,
                                              type_delta,
                                              act1,
                                              act2,
                                              list_totalindices)
                 

             list_dict_actualdensitities_meth.append(dict_actual_density_meth)
            
         dict_results[key]= {"Total":dict_total_actualdensity,
                            "Each impact category":list_dict_actualdensitities_meth}    
            
            
     return dict_results      
    
    
    
    


def extract_compa_delta_main(input_string):
    # Assuming a string structure that allows for this function to extract comparison and delta values
    compa_match = re.search(r'compa_(.*?)tresh', input_string)
    delta_match = re.search(r'delta_(-?\d+(\.\d+)?)', input_string)
    
    if compa_match and delta_match:
        compa = compa_match.group(1)
        delta_value = float(delta_match.group(1))
        return compa, delta_value
    else:
        raise ValueError("No compa or delta value found in the input string.")

def extract_meth_accronym(input_string):
    # Define the regular expression pattern to match any characters between parentheses
    pattern = r"\((.*?)\)"
    match = re.search(pattern, input_string)
    
    meth_accronym = match.group(1)


    return meth_accronym
     









#####
""" Setting up"""
#####




# Same impact categoris as for the original boxes. (Could be immported isntead to avoid mistakes.)

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





bd.projects.set_current('AVS')




Ecoinvent = bd.Database('ecoinvent-3.10-consequential')

biosphere = bd.Database('ecoinvent-3.10-biosphere')


foregroundAVS = bd.Database("foregroundAVS")




# Create a parameterized marginal mix according to the choice of meths
additional_biosphere_multi_categories,list_modif_meth,list_virtual_emissions_same_order_as_meth =  add_virtual_flow_to_additional_biosphere(list_meth)
elec_marginal_fr_copy = init_act_variation_impact(elec_marginal_fr_copy,list_meth,additional_biosphere_multi_categories)


# dict_funct imported from Parameters_and_functions

dict_funct["modif_impact_marginal_elec"]={"func":vectorize_function(modif_impact_marginal_elec),
                     "indices":[(act_meth.id,elec_marginal_fr_copy.id) for act_meth in list_virtual_emissions_same_order_as_meth]}



# Collect all indices/positions in the LCA matrixes on which functions will apply
list_totalindices = []                                 
for a in dict_funct:
    
    list_totalindices= list_totalindices+dict_funct[a]["indices"]





###
"""Choice of settings"""
###

ray.shutdown()

ray.init()

#num_cpus = int(ray.available_resources()["CPU"])
num_cpus = 10 
 
    
main="elec" # Main product as chosen for the original boxes. Can be changed to "both" or "crop".
size=2500 # Number of stochastic computations to check the actual density of a box.

uncert= False # Do not change
sampling_rate_boxes =10 # Every 10th box is double-checked.


# Do not change
type_delta="percent_modif" # Type of comparison: percentage of modification (A X% better or worse than B). Do not change. 
act1="AVS_elec_main" # Comparison tested = act 1 - act 2. Do not change
act2="PV_ref" # Comparison tested = act 1 - act 2. Do not change



###
"""Running and exporting results"""
###



time1 = time.time()

dict_res_checkingboxes = check_boxes_of_dict(dictres_import_elec_varpv,
                        main,
                        size,
                        list_meth,
                        sampling_rate_boxes,
                        dictionnaries,
                        meth1,
                        uncert,
                        dict_funct,
                        num_cpus,
                        elec_marginal_fr_copy,
                        Ecoinvent,
                        biosphere,
                        foregroundAVS,
                        type_delta,
                        act1,
                        act2,
                        list_totalindices)  



timetot=time.time()-time1

print("Total time to check the whole dict:",timetot)

x = datetime.datetime.now()

month=str(x.month)
day=str(x.day)
microsec=str(x.strftime("%f"))


name_file_res ='checkboxes'+"_"+main+month+"_"+day+"_"+microsec+"size="+str(size)




export_pickle_2(dict_res_checkingboxes, name_file_res, "PRIM_process/Boxes")




