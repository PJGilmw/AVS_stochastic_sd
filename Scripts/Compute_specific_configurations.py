# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:43:51 2024

@author: pjouannais

Computes LCA scores for chosen AVS configurations and their conventional counterparts.
Execute the whole script. 
The tested configurations can be changed in the parameters dictionnaries l312

"""




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
    
from Main_functions  import *
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

import re


import seaborn as sns
import matplotlib.pyplot as plt




###
"""Additional functions to organize results before plotting"""
###


def split_at_comma(input_string):
    """
    Splits the input string into two parts at the first comma.
    
    Parameters:
    input_string (str): The string to split.
    
    Returns:
    tuple: A tuple containing the substring before the comma and the substring after the comma.
           If no comma is found, the second element of the tuple will be an empty string.
    """
    # Find the position of the first comma
    comma_position = input_string.find(',')
    
    # If no comma is found, return the whole string as the first part and an empty string as the second part
    if comma_position == -1:
        return (input_string, "")
    
    # Split the string at the comma position
    before_comma = input_string[:comma_position]
    after_comma = input_string[comma_position + 1:]
    
    return before_comma, after_comma

def crop_type(row):
    """ Collects and returns the crop type of a dataframe's row (when crop is main product)"""
    
    value =  row["FU"]
    print(value)
    if "wheat" in value or "Wheat" in value:
        print("ok")
        new_value="Wheat"
        
    elif "soy" in value or "Soy" in value:
        new_value="Soy"
        
    elif "alfalfa" in value or "Alfalfa" in value:
        
        new_value="Alfalfa"           
    
    return new_value 

def crop_type_elec(row):
    
    """ Collects and returns the crop type of a dataframe's row (when elec is main product)"""


    value =  row["FU"]
    print(value)
    if "wheat" in value or "Wheat" in value:
        print("ok")
        new_value="Wheat"
        
    elif "soy" in value or "Soy" in value:
        new_value="Soy"
        
    elif "alfalfa" in value or "Alfalfa" in value:
        
        new_value="Alfalfa" 
    else:
        new_value = " "          
    
    return new_value 

def AVS_type(row):
    
    """ Returns"" is the row is for AVS or for conventional (when crop is main product) """

    value =  row["FU"]
    print(value)
    if "AVS" in value :
        print("ok")
        new_value="AVS"
        
    else: 
        new_value="Ref."
        

    return new_value 

def AVS_type_elec(row):
    
    """ Returns"" is the row is for AVS or for conventional (when electricity is main product) """


    value =  row["FU"]
    print(value)
    if "AVS" in value :
        print("ok")
        new_value="AVS"
        
    elif "PV" in value: 
        new_value="PV"
        
    else: 
        new_value="Marginal"
                

    return new_value 


def extract_parentheses_content(s):
    """
    Extracts the content between the first pair of parentheses in the string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The content between the first pair of parentheses, or an empty string if no parentheses are found.
    """
    match = re.search(r'\((.*?)\)', s)
    return match.group(1) if match else ''



def reset_unique(dict_):
    
    for a in dict_:
        
        if dict_[a][0]=="unif":
            
            dict_[a][0]="unique"
            
            
    return(dict_)        


def find_crop_names(row,colnames_switch):
    

    value_cropswitch=None
    for colname in colnames_switch:
        #print(row[colname])

        value =  row[colname]
        if value==1:
            value_cropswitch=colname
            
    
    return value_cropswitch

def melt_switch_names(sample_dataframe,colnames_switch,name_meltcolumn):
    
    sample_dataframe_melt = sample_dataframe.copy()
    sample_dataframe_melt[name_meltcolumn]= [find_crop_names(sample_dataframe.iloc[i],colnames_switch) for i in range(len(sample_dataframe))]
    sample_dataframe_melt.drop(colnames_switch, axis=1, inplace=True)

    return sample_dataframe_melt








"""Run LCAs"""


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







""" Load project, db, methods etc."""

# Check if needed or if it's imported with the other scripts
bd.projects.set_current('AVS')


Ecoinvent = bd.Database('ecoinvent-3.10-consequential')

biosphere = bd.Database('ecoinvent-3.10-biosphere')


foregroundAVS = bd.Database("foregroundAVS")

# Create a parameterized marginal elec according to the choice of meths
additional_biosphere_multi_categories,list_modif_meth,list_virtual_emissions_same_order_as_meth =  add_virtual_flow_to_additional_biosphere(list_meth)




elec_marginal_fr_copy = init_act_variation_impact(elec_marginal_fr_copy,list_meth,additional_biosphere_multi_categories)
#init_act_variation_impact(act_ori,act_copy,list_meth,additional_biosphere_multi_categories)

dict_funct["modif_impact_marginal_elec"]={"func":vectorize_function(modif_impact_marginal_elec),
                     "indices":[(act_meth.id,elec_marginal_fr_copy.id) for act_meth in list_virtual_emissions_same_order_as_meth]}

list_totalindices = []                                 
for a in dict_funct:
    
    list_totalindices= list_totalindices+dict_funct[a]["indices"]








"""Definitions of the tested configurations (scenarios)"""



Scenarios_to_compute={}

# PV updated by Besseau + NO iLUC +  0 interaction effect
# BASELINE

# Initialize the dictionnaries

# Only copy paste of the original dictionnaires, only the "unique" values will be kept.

iluc_par_distributions = {

    "NPP_weighted_ha_y_eq_cropref":['unif', [1.16,0.5, 2, 0, 0],"."],   # France
    "NPP_weighted_ha_y_eq_ratio_avs_cropref":['unif', [1,0.5, 2, 0, 0],"."],
    "NPP_weighted_ha_y_eq_ratio_pv_cropref":['unif', [1,0.5, 2, 0, 0],"."],
    "iluc_lm_PV":['unif', [0,300, 2200, 0, 0],"kg CO2 eq.ha eq-1"],        # Global Arable land
    "iluc_lm_AVS":['unif', [0,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"], # Global Arable land
    "iluc_lm_cropref":['unif', [0,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"] # Global Arable land

    
    }

# Original yield in ecoinvent
output_wheat = [exc["amount"] for exc in list(wheat_fr_ref.exchanges()) if exc["type"]=="production"][0]
output_soy = [exc["amount"] for exc in list(soy_ch_ref.exchanges()) if exc["type"]=="production"][0]
output_alfalfa = [exc["amount"] for exc in list(alfalfa_ch_ref.exchanges()) if exc["type"]=="production"][0]


Ag_par_distributions = {

    "prob_switch_crop":['switch',3,[1/3,1/3,1/3],["wheat","soy","alfalfa"],"."],
    "crop_yield_upd_ref":['unif', [1,0.5, 2, 0, 0],"."],
    "crop_yield_ratio_avs_ref":['unif', [1,0.3, 2, 0, 0],"."],
    "crop_fert_upd_ref":['unif', [1,0.5, 2, 0, 0],"."],
    "crop_fert_ratio_avs_ref":['unif', [1,0.3, 2, 0, 0],"."],
    "crop_mach_upd_ref":['unif', [1,0.5, 2, 0, 0],"."],
    "crop_mach_ratio_avs_ref":['unif', [1,0.3, 2, 0, 0],"."],
    "water_upd_ref":['unif', [1,0.5, 2, 0, 0],"."],
    "water_ratio_avs_ref":['unif', [1,0.3, 2, 0, 0],"."],
    "carbon_accumulation_soil_ref":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
    "carbon_accumulation_soil_AVS":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
    "carbon_accumulation_soil_PV":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
    
    "init_yield_alfalfa":['unique', [output_alfalfa,0, 0, 0, 0],"."],
    "init_yield_soy":['unique', [output_soy,0, 0, 0, 0],"."],
    "init_yield_wheat":['unique', [output_wheat,0, 0, 0, 0],"."]
    }
    
    


PV_par_distributions = {
    'annual_producible_PV': ['unif', [1200, 900, 2000, 0, 0],"kwh.kwp-1.y-1"],
    'annual_producible_ratio_avs_pv': ['unif', [1,0.6, 1.3, 0, 0],"."],
    
    'mass_to_power_ratio_electric_instal_PV': ['unif', [2.2,1.5, 8, 0, 0],"kg electric installation . kwp -1"],  # Besseau minimum 2.2
    
    'panel_efficiency_PV': ['unif', [0.228,0.10, 0.40, 0, 0],"."],   # Besseau maximum 22.8%
    
    
    'inverter_specific_weight_PV': ['unif', [0.85,0.4, 7, 0, 0],"kg inverter.kwp-1"],   # Besseau minimum 0.85
   
    'inverter_lifetime_PV': ['unif', [15,10, 30, 0, 0],"y"], # Besseau 15

    'plant_lifetime_PV': ['unif', [30,20, 40, 0, 0],"y"], # Besseau and ecoinvent 30
    
    'plant_lifetime_ratio_avs_pv': ['unique', [1,0.3, 2, 0, 0],"."],  
    
    'concrete_mount_upd_PV': ['unif', [1,0.1, 2, 0, 0],"."],
    
    'concrete_mount_ratio_avs_pv': ['unif', [1,0.1, 4, 0, 0],"."],   # here facotr 4 because could be way worse
    
    'aluminium_mount_upd_PV': ['unif', [0.38,0.1, 2, 0, 0],"."], # here besseau suggests 1.5 kilo and the original value is 3.98. So update = 1.5/3.98 =  0.38
        
    'aluminium_mount_ratio_avs_pv': ['unif', [1,0.1, 4, 0, 0],"."],
    
    'steel_mount_upd_PV': ['unif', [0.3,0.1, 2, 0, 0],"."],
        
    'steel_mount_ratio_avs_pv': ['unif', [1,0.1, 4, 0, 0],"."],

    'poly_mount_upd_PV': ['unif', [0.3,0.1, 2, 0, 0],"."],
        
    'poly_mount_ratio_avs_pv': ['unif', [1,0.1,4 , 0, 0],"."],


    'corrbox_mount_upd_PV': ['unif', [0.3,0.1, 2, 0, 0],"."],
        
    'corrbox_mount_ratio_avs_pv': ['unif', [1,0.1, 4, 0, 0],"."],
    
    
    'zinc_mount_upd_PV': ['unif', [0.38,0.1, 2, 0, 0],"."],
        
    'zinc_mount_ratio_avs_pv': ['unif', [1,0.1, 4, 0, 0],"."],


    'surface_cover_fraction_PV': ['unif', [0.4,0.20, 0.6, 0, 0],"m2panel.m-2"],
    
    'surface_cover_fraction_AVS': ['unif', [0.4,0.10, 0.6, 0, 0],"."],
    
    
    
    
    'aluminium_panel_weight_PV': ['unif', [1.5,0, 3, 0, 0],"kg aluminium.m-2 panel"],  #Besseau :The ecoinvent inventory model assumes 2.6 kg aluminum/m2 whereas more recent studies indicate 1.5 kg/m2.42 Some PV panels are even frameless
    
    


    'manufacturing_eff_wafer_upd_PV': ['unif', [1,0.1, 2, 0, 0],"."],

    'solargrade_electric_intensity_PV': ['unif', [30,20, 110, 0, 0],"."],
    

    "prob_switch_substielec":['switch',2,[1,0],["substit_margi","substit_PV"],"."],
    
    "impact_update_margi":['unif',[1,0.01, 3, 0, 0],"."]

    }







# "reset" sets all "unif" distributions to "unique" values
dictionnaries = {"Ag_par_distributions" : reset_unique(Ag_par_distributions),
                 "PV_par_distributions": reset_unique(PV_par_distributions),
                 "iluc_par_distributions":reset_unique(iluc_par_distributions)}




Scenarios_to_compute["No effect, no iLUC"]=dictionnaries




# PV updated + iLUC +  NO effect
# We only indicate the values that change compared to the orginal dictionnary 

iluc_par_distributions_pvup_iluc_noeffect = iluc_par_distributions.copy()

iluc_par_distributions_pvup_iluc_noeffect["iluc_lm_PV"]=['unique', [1840,0.3, 2, 0, 0],"."]   
iluc_par_distributions_pvup_iluc_noeffect["iluc_lm_AVS"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_noeffect["iluc_lm_cropref"]=['unique', [1840,0.3, 2, 0, 0],"."]



dictionnaries_pvup_iluc_noeffect = {"Ag_par_distributions" : reset_unique(Ag_par_distributions),
                 "PV_par_distributions": reset_unique(PV_par_distributions),
                 "iluc_par_distributions":reset_unique(iluc_par_distributions_pvup_iluc_noeffect)}


Scenarios_to_compute["No effect, iLUC"]=dictionnaries_pvup_iluc_noeffect









# PV updated + iLUC +  negative interaction effect on elec: 15% decrease yield on crop 

Ag_par_distributions_pvup_iluc_neg_crop = Ag_par_distributions.copy()
PV_par_distributions_pvup_iluc_neg_crop = PV_par_distributions.copy()
iluc_par_distributions_pvup_iluc_neg_crop = iluc_par_distributions.copy()

iluc_par_distributions_pvup_iluc_neg_crop["iluc_lm_PV"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_neg_crop["iluc_lm_AVS"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_neg_crop["iluc_lm_cropref"]=['unique', [1840,0.3, 2, 0, 0],"."]



Ag_par_distributions_pvup_iluc_neg_crop["crop_yield_ratio_avs_ref"]=['unique', [0.85,0.3, 2, 0, 0],"."]





dictionnaries_pvup_iluc_neg_crop = {"Ag_par_distributions" : reset_unique(Ag_par_distributions_pvup_iluc_neg_crop),
                 "PV_par_distributions": reset_unique(PV_par_distributions_pvup_iluc_neg_crop),
                 "iluc_par_distributions":reset_unique(iluc_par_distributions_pvup_iluc_neg_crop)}





Scenarios_to_compute["-15% crop yield, iLUC"]=dictionnaries_pvup_iluc_neg_crop



# PV updated + iLUC +  negative interaction effect on elec: 15% decrease yield elec  

Ag_par_distributions_pvup_iluc_neg_elec = Ag_par_distributions.copy()
PV_par_distributions_pvup_iluc_neg_elec = PV_par_distributions.copy()
iluc_par_distributions_pvup_iluc_neg_elec = iluc_par_distributions.copy()

iluc_par_distributions_pvup_iluc_neg_elec["iluc_lm_PV"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_neg_elec["iluc_lm_AVS"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_neg_elec["iluc_lm_cropref"]=['unique', [1840,0.3, 2, 0, 0],"."]



PV_par_distributions_pvup_iluc_neg_elec["annual_producible_ratio_avs_pv"]=['unique', [0.85,0.3, 2, 0, 0],"."]







dictionnaries_pvup_iluc_neg_elec = {"Ag_par_distributions" : reset_unique(Ag_par_distributions_pvup_iluc_neg_elec),
                 "PV_par_distributions": reset_unique(PV_par_distributions_pvup_iluc_neg_elec),
                 "iluc_par_distributions":reset_unique(iluc_par_distributions_pvup_iluc_neg_elec)}





Scenarios_to_compute["-15% electricity yield, iLUC"]=dictionnaries_pvup_iluc_neg_elec




# PV updated +  postitive effect elec: 15% increase yield elec 

Ag_par_distributions_pvup_iluc_pos_elec = Ag_par_distributions.copy()
PV_par_distributions_pvup_iluc_pos_elec = PV_par_distributions.copy()
iluc_par_distributions_pvup_iluc_pos_elec = iluc_par_distributions.copy()



iluc_par_distributions_pvup_iluc_pos_elec["iluc_lm_PV"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_pos_elec["iluc_lm_AVS"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_pos_elec["iluc_lm_cropref"]=['unique', [1840,0.3, 2, 0, 0],"."]



PV_par_distributions_pvup_iluc_pos_elec["annual_producible_ratio_avs_pv"]=['unique', [1.15,0.3, 2, 0, 0],"."]





dictionnaries_pvup_iluc_pos_elec = {"Ag_par_distributions" : reset_unique(Ag_par_distributions_pvup_iluc_pos_elec),
                 "PV_par_distributions": reset_unique(PV_par_distributions_pvup_iluc_pos_elec),
                 "iluc_par_distributions":reset_unique(iluc_par_distributions_pvup_iluc_pos_elec)}


Scenarios_to_compute["+15% electricity yield, iLUC"]=dictionnaries_pvup_iluc_pos_elec


# PV updated +  postitive effect crop: 15% increase yield crop 

Ag_par_distributions_pvup_iluc_pos_crop = Ag_par_distributions.copy()
PV_par_distributions_pvup_iluc_pos_crop = PV_par_distributions.copy()
iluc_par_distributions_pvup_iluc_pos_crop = iluc_par_distributions.copy()




Ag_par_distributions_pvup_iluc_pos_crop["crop_yield_ratio_avs_ref"]=['unique', [1.15,0.3, 2, 0, 0],"."]



iluc_par_distributions_pvup_iluc_pos_crop["iluc_lm_PV"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_pos_crop["iluc_lm_AVS"]=['unique', [1840,0.3, 2, 0, 0],"."]
iluc_par_distributions_pvup_iluc_pos_crop["iluc_lm_cropref"]=['unique', [1840,0.3, 2, 0, 0],"."]



dictionnaries_pvup_iluc_pos_crop = {"Ag_par_distributions" : reset_unique(Ag_par_distributions_pvup_iluc_pos_crop),
                 "PV_par_distributions": reset_unique(PV_par_distributions_pvup_iluc_pos_crop),
                 "iluc_par_distributions":reset_unique(iluc_par_distributions_pvup_iluc_pos_crop)}


Scenarios_to_compute["+15% crop yield, iLUC"]=dictionnaries_pvup_iluc_pos_crop








### 
"""RUN """
#It's not convenient to fix the crop type manually so instead, we keep a 1/3 proba for each crop type and compute stochastic combinations to make sure that we obtain each crop type simumated for each scenario.
# We collect them afterwards.

colnames_switch=['wheat_switch', 'soy_switch','alfalfa_switch']

uncert=False

num_cpus =5

size=16 # Just to be sure to have at least the 3 crops
list_tablesmelt = []

list_sample =[]
for key_name in Scenarios_to_compute:
    
    print(key_name)
    
    dictionnaries = Scenarios_to_compute[key_name]

    
    names_param_total, sample_dataframe= sampling_func_lhs(dictionnaries,
                                 size)
    
    list_sample.append(sample_dataframe)
    contains_onewheat = sample_dataframe['wheat_switch'].eq(1).any()
    contains_onesoy = sample_dataframe['soy_switch'].eq(1).any()
    contains_onealfalfa = sample_dataframe['alfalfa_switch'].eq(1).any()

    if contains_onewheat and contains_onesoy and contains_onealfalfa:
        print("OK  switch")
    else:
        sys.exit("Stopping execution because condition was met.")

        
    
    
    values_for_datapackages={}
    for dict in dictionnaries:
       values_for_datapackages = Merge(values_for_datapackages , dictionnaries[dict])
        
       
    values_for_datapackages= fix_switch_valuedatapackages(values_for_datapackages)
    
    for param in sample_dataframe:
        
        values_for_datapackages[param]={"values":sample_dataframe[param].tolist()}
    
    
    
    
    """Create the modified datapackage"""
    
    
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




    for meth_index in range(len(list_tables)):
       

        table_imp = list_tables[meth_index]

        table_imp_modif = pd.concat([sample_dataframe,table_imp],axis=1)
        table_imp_modif = table_imp_modif.drop_duplicates(subset=['wheat_switch', 'soy_switch','alfalfa_switch'])
        table_imp_modif.reset_index(drop=True, inplace=True)
         
        table_imp_modif=melt_switch_names(table_imp_modif,colnames_switch,"crop")
        
        table_imp_modif=table_imp_modif[np.concatenate([["crop"],table_imp.columns]).tolist()]
        
        #table_imp_modif["index"]=table_imp_modif.index.tolist()

        #table_imp_modif=pd.melt(table_imp_modif, id_vars=["index","crop"], value_vars=table_imp_modif.columns.tolist())
        table_imp_modif=pd.melt(table_imp_modif, id_vars=["crop"], value_vars=table_imp_modif.columns.tolist())
        
        
        # Remove duplicates for the Fu which are not affected by the choice of crop
        
        table_imp_modif_sub1 = table_imp_modif[table_imp_modif["variable"].isin(["wheat_fr_ref", "soy_ch_ref", "alfalfa_ch_ref", "elec_marginal_fr_copy","PV_ref"])]
        table_imp_modif_sub1 = table_imp_modif_sub1.drop_duplicates(subset=['value', 'variable'])
        table_imp_modif_sub1["crop"] = ""

        table_imp_modif_sub2 = table_imp_modif[~table_imp_modif["variable"].isin(["wheat_fr_ref", "soy_ch_ref", "alfalfa_ch_ref", "elec_marginal_fr_copy","PV_ref"])]

        # Reunify table after the different treatments
        table_imp_modif_treated = pd.concat([table_imp_modif_sub1,table_imp_modif_sub2])
        
        
        table_imp_modif_treated["Impact Category"] = list_meth[meth_index][-1]
    
        table_imp_modif_treated["Scenario"] = key_name
        
        list_tablesmelt.append(table_imp_modif_treated)
        
        
        
        
        
        
        
###        
"""Reorganize results"""
###


        
recombine_table = pd.concat(list_tablesmelt,axis=0)
# recombine_table = recombine_table.drop_duplicates(subset=['Impact Category', 'Scenario','alfalfa_switch'])

# recombine_table = recombine_table[recombine_table.variable != "elec_marginal_fr_copy"]
    
recombine_table["FU"] = recombine_table["crop"] + recombine_table["variable"]


for meth in list_meth:
    
    recombine_table['Impact Category'] = recombine_table['Impact Category'].replace(meth[-1], extract_parentheses_content(meth[-1]))


recombine_table['FU'] = recombine_table['FU'].replace("PV_ref", "PV")
recombine_table['FU'] = recombine_table['FU'].replace("wheat_fr_ref", "Wheat ref.")
recombine_table['FU'] = recombine_table['FU'].replace("soy_ch_ref", "Soy ref.")
recombine_table['FU'] = recombine_table['FU'].replace("alfalfa_ch_ref", "alfalfa ref.")

recombine_table['FU'] = recombine_table['FU'].replace("elec_marginal_fr_copy", "Marginal electricity")
recombine_table['FU'] = recombine_table['FU'].replace("soy_switchAVS_elec_main", "AVS electricity, soy")
recombine_table['FU'] = recombine_table['FU'].replace("wheat_switchAVS_elec_main", "AVS electricity, wheat")
recombine_table['FU'] = recombine_table['FU'].replace("alfalfa_switchAVS_elec_main", "AVS electricity, alfalfa")

recombine_table['FU'] = recombine_table['FU'].replace("soy_switchAVS_crop_main", "AVS crop, soy")
recombine_table['FU'] = recombine_table['FU'].replace("wheat_switchAVS_crop_main", "AVS crop, wheat")
recombine_table['FU'] = recombine_table['FU'].replace("alfalfa_switchAVS_crop_main", "AVS crop, alfalfa")





# Make two plots, separates electricy FU and crop FU
recombine_table_elec_FU = recombine_table[recombine_table["FU"].isin(["PV", "Marginal electricity", "AVS electricity, alfalfa", "AVS electricity, wheat","AVS electricity, soy"])]
recombine_table_crop_FU = recombine_table[~recombine_table["FU"].isin(["PV", "Marginal electricity", "AVS electricity, alfalfa", "AVS electricity, wheat","AVS electricity, soy"])]



sns.set_style("darkgrid")

# Organize results before plotting

recombine_table_elec_FU['FU'] = recombine_table_elec_FU['FU'].replace("PV electricity")

recombine_table_elec_FU['Impact Category'] = recombine_table_elec_FU['Impact Category'].replace("FEP", "FE,kg P-eq.")
recombine_table_elec_FU['Impact Category'] = recombine_table_elec_FU['Impact Category'].replace("TETP", "TETinf,kg 1.4-DCB.")
recombine_table_elec_FU['Impact Category'] = recombine_table_elec_FU['Impact Category'].replace("GWP100", "GW100,kg CO2-eq.")
recombine_table_elec_FU['Impact Category'] = recombine_table_elec_FU['Impact Category'].replace("LOP", "LO,m2*a crop-eq.")
recombine_table_elec_FU['Impact Category'] = recombine_table_elec_FU['Impact Category'].replace("SOP", "SO,kg Cu-eq.")
recombine_table_elec_FU['Impact Category'] = recombine_table_elec_FU['Impact Category'].replace("PMFP", "PMF,kg PM2.5-eq.")
recombine_table_elec_FU['Impact Category'] = recombine_table_elec_FU['Impact Category'].replace("WCP", "WC,m3")

recombine_table_crop_FU['Impact Category'] = recombine_table_crop_FU['Impact Category'].replace("FEP", "FE,kg P-eq.")
recombine_table_crop_FU['Impact Category'] = recombine_table_crop_FU['Impact Category'].replace("TETP", "TETinf,kg 1.4-DCB-eq.")
recombine_table_crop_FU['Impact Category'] = recombine_table_crop_FU['Impact Category'].replace("GWP100", "GW100,kg CO2-eq.")
recombine_table_crop_FU['Impact Category'] = recombine_table_crop_FU['Impact Category'].replace("LOP", "LO,m2*a crop-eq.")
recombine_table_crop_FU['Impact Category'] = recombine_table_crop_FU['Impact Category'].replace("SOP", "SO,kg Cu-eq.")
recombine_table_crop_FU['Impact Category'] = recombine_table_crop_FU['Impact Category'].replace("PMFP", "PMF,kg PM2.5-eq.")
recombine_table_crop_FU['Impact Category'] = recombine_table_crop_FU['Impact Category'].replace("WCP", "WC,m3")



recombine_table_crop_FU["Crop"]= [crop_type(recombine_table_crop_FU.iloc[i])for i in range(len(recombine_table_crop_FU))]
recombine_table_crop_FU["Type"]= [AVS_type(recombine_table_crop_FU.iloc[i])for i in range(len(recombine_table_crop_FU))]

recombine_table_elec_FU["Crop"]= [crop_type_elec(recombine_table_elec_FU.iloc[i])for i in range(len(recombine_table_elec_FU))]
recombine_table_elec_FU["Type"]= [AVS_type(recombine_table_elec_FU.iloc[i])for i in range(len(recombine_table_elec_FU))]



    
    
# Crop



recombine_table_crop_FU_sub_ref =  recombine_table_crop_FU[recombine_table_crop_FU["Type"]=="Ref."]

rows_to_drop = recombine_table_crop_FU[recombine_table_crop_FU["Type"] == "Ref."]
# Then, drop these rows using their index
recombine_table_crop_FU_part1 = recombine_table_crop_FU.drop(rows_to_drop.index)


recombine_table_crop_FU_sub_ref = recombine_table_crop_FU_sub_ref[recombine_table_crop_FU_sub_ref["Scenario"].isin(["No effect, no iLUC", "No effect, iLUC"])]

recombine_table_crop_FU_sub_ref = recombine_table_crop_FU_sub_ref.drop_duplicates(subset=['variable', 'Impact Category','Scenario'])


recombine_table_crop_FU_sub_ref['Scenario'] = recombine_table_crop_FU_sub_ref["Scenario"].replace("No effect, no iLUC", "Conventional, No iLUC")
recombine_table_crop_FU_sub_ref['Scenario'] = recombine_table_crop_FU_sub_ref["Scenario"].replace("No effect, iLUC", "Conventional, iLUC")

recombine_table_crop_FU_fixed = pd.concat([recombine_table_crop_FU_part1,
                                           recombine_table_crop_FU_sub_ref])






# Elec



recombine_table_elec_FU_sub_ref =  recombine_table_elec_FU[recombine_table_elec_FU["Type"]=="Ref."]

rows_to_drop = recombine_table_elec_FU[recombine_table_elec_FU["Type"] == "Ref."]
# Then, drop these rows using their index
recombine_table_elec_FU_part1 = recombine_table_elec_FU.drop(rows_to_drop.index)


recombine_table_elec_FU_sub_ref = recombine_table_elec_FU_sub_ref[recombine_table_elec_FU_sub_ref["Scenario"].isin(["No effect, no iLUC", "No effect, iLUC"])]

recombine_table_elec_FU_sub_ref = recombine_table_elec_FU_sub_ref.drop_duplicates(subset=['variable', 'Impact Category','Scenario'])


recombine_table_elec_FU_fixed = pd.concat([recombine_table_elec_FU_part1,
                                           recombine_table_elec_FU_sub_ref])







# Custom colors
custom_palette = {
    "+15% crop yield, iLUC": "#aec7e8",  # light blue
    "15% electricity yield, iLUC": "#ffbb78",  # light orange
    "-15% electricity yield, iLUC": "#98df8a",  # light green
    "Conventional, No iLUC": "#1a2421",  # dark
    "Conventional, iLUC": "#6b6b6b",  # dark grey
    "No effect, iLUC": "#c7c7c7",  # light gray
    "No effect, no iLUC": "#ffd966",  # light blue
    "-15% crop yield, iLUC": "#ae8d81",  # light brown
    '+15% electricity yield, iLUC': "#ffbcd9"  # light pink
}



  



"""Plots for crop as a main product"""


# Crop plots
# Get unique values of the 'Impact Category' column
impact_categories_crop = recombine_table_crop_FU_fixed['Impact Category'].unique()


for category in impact_categories_crop:
    
    # Filter the DataFrame for the current category
    subset = recombine_table_crop_FU_fixed[recombine_table_crop_FU_fixed['Impact Category'] == category]
    
    category, unit = split_at_comma(category)
    print(category)
    print(unit)

    if category == "GW100":
        unit = "kg $CO_{2}$-eq"
        
    elif category =="WC":
        unit = "$m^{3}$-eq"

        
    
    g = sns.FacetGrid(subset, row="Crop", margin_titles=True, height=2, aspect=4, sharey=True)
    
    # Map a barplot onto each facet
    g.map_dataframe(sns.barplot, y="value", hue="Scenario", palette=custom_palette, ci=None)
    
    # Rotate the x-axis labels and increase font size
    for ax in g.axes.flatten():
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=12)  # Adjust fontsize as needed
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Set font size for x-axis labels
        ax.set_ylabel(unit, fontsize=12)  # Set y-axis label to the value of 'unit'
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # Increase the number of y-axis ticks
    
    # Increase the size of the titles for the rows and columns
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=14)
    
    # Add a main title for the entire figure with the impact category
    plt.subplots_adjust(top=0.9)  # Adjust the top margin to make room for the title
    g.fig.suptitle(f'{category}', fontsize=16)
    
    # Move the legend below the plot
    g.add_legend(title='Scenario', bbox_to_anchor=(0.5, -0.1), loc='center', ncol=2, fontsize=12)
    
    # Adjust layout to add space between subplots and for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot to a file in the local directory
    plt.savefig(f"plot_crop_{category}_fixed.png", bbox_inches='tight', dpi=600)
    
    # Show plot
    plt.show()
    
    # Clear the current plot to avoid overlap
    plt.clf()
    plt.close()






"""Plots for electricity as a main product"""
# Get unique values of the 'Impact Category' column
impact_categories_elec = recombine_table_elec_FU_fixed['Impact Category'].unique()

# Loop over each unique value of 'Impact Category' and create separate plots
for category in impact_categories_elec:
    # Filter the DataFrame for the current category
    subset = recombine_table_elec_FU_fixed[recombine_table_elec_FU_fixed['Impact Category'] == category]
    
    category, unit = split_at_comma(category)
    print(category)
    print(unit)

    if category == "GW100":
        unit = "kg $CO_{2}$-eq"
        
    elif category =="WC":
        unit = "$m^{3}$-eq"

       
    # Create a new figure with the same dimensions as the FacetGrid
    plt.figure(figsize=(12, 6), dpi=500)
    
    # Create the barplot
    ax = sns.barplot(data=subset, x="FU", y="value", hue="Scenario", palette=custom_palette, ci=None,legend=False)
    
    # Rotate the x-axis labels and increase font size
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=12)  # Adjust fontsize as needed
    ax.set_xlabel('FU', fontsize=0)  # Set font size for x-axis labels
    ax.set_ylabel(unit, fontsize=14)  # Set font size for y-axis labels
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # Increase the number of y-axis ticks
    
    # Add a main title for the entire figure with the impact category
    plt.title(f'{category}', fontsize=16)
    
    # Move the legend below the plot
    #plt.legend(title='Scenario', bbox_to_anchor=(0.5, -0.3), loc='center', ncol=2, fontsize=12)
    
    # Adjust layout to add space between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    category_plot = category[:2]

    # Save the plot to a file in the local directory
    plt.savefig(f"plot_elec_{category_plot}_fixed.png", bbox_inches='tight', dpi=600)
    
    # Show plot
    plt.show()
    
    # Clear the current plot to avoid overlap
    plt.clf()
    plt.close()
