# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:35:49 2024

@author: pierre.jouannais

Script containing the definition of the parameters and the functions modifying the amounts in the LCA matrixes.
The parameters value can be modified from line52
"""


import pandas as pd
import decimal
from random import *
import pstats
from itertools import *
from math import*
import csv
import copy
import numpy as np
import random


import math


import Activities

from Activities import *


import inspect

from scipy.stats.qmc import LatinHypercube



import bw_processing as bwp

#########################################
#########################################
"""Parameters definitions"""
##########################################
#########################################"



# Parameters distributions dictionnaries

#
# each parameter is assigned a list containing  :
   # [Distribution,unique value if unique, min,max,mode,sd, unit]

# Distribution :
#   - 'unique' if no distribution. The  algorithm considers the value "unique"
#   - 'unif' for uniform, uses min and max
#   - 'triang, uses mim max and mode with mode as a fracion of max-min

#


# iLUC param
iluc_par_distributions = {

    "NPP_weighted_ha_y_eq_cropref":['unif', [1,0.5, 2, 0, 0],"."],
    "NPP_weighted_ha_y_eq_ratio_avs_cropref":['unif', [1,0.5, 2, 0, 0],"."],
    "NPP_weighted_ha_y_eq_ratio_pv_cropref":['unif', [1,0.5, 2, 0, 0],"."],
    "iluc_lm_PV":['unif', [1800,300, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
    "iluc_lm_AVS":['unif', [300,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
    "iluc_lm_cropref":['unif', [1800,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"]

    
    }

# We collect the intitial output values for the crop activities and use them as parameters


output_wheat = [exc["amount"] for exc in list(wheat_fr_ref.exchanges()) if exc["type"]=="production"][0]
output_soy = [exc["amount"] for exc in list(soy_ch_ref.exchanges()) if exc["type"]=="production"][0]
output_alfalfa = [exc["amount"] for exc in list(alfalfa_ch_ref.exchanges()) if exc["type"]=="production"][0]

# Agronomical param
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
    
    

# PV param
PV_par_distributions = {
    'annual_producible_PV': ['unif', [1200, 900, 2000, 0, 0],"kwh.kwp-1.y-1"],
    'annual_producible_ratio_avs_pv': ['unif', [1,0.6, 1.3, 0, 0],"."],
    
    'mass_to_power_ratio_electric_instal_PV': ['unif', [2.2,1.5, 8, 0, 0],"kg electric installation . kwp -1"],  # Besseau minimum 2.2
    
    'panel_efficiency_PV': ['unif', [0.228,0.10, 0.40, 0, 0],"."],   # Besseau maximum 22.8%
    
    
    'inverter_specific_weight_PV': ['unif', [0.85,0.4, 7, 0, 0],"kg inverter.kwp-1"],   # Besseau maximum 0.85
   
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



def reset_unique(dict_):
    """Function that sets all the parameters of dict_ to their unique values."""
    
    for a in dict_:
        
        if dict_[a][0]=="unif":
            
            dict_[a][0]="unique"
            
            
    return(dict_)        





# Put all the dictionnaries in one

dictionnaries = {"Ag_par_distributions" : Ag_par_distributions,
                 "PV_par_distributions": PV_par_distributions,
                 "iluc_par_distributions":iluc_par_distributions}






####################
"""Define functions"""
######################



#######
"""Accessory/computation functions"""
#######
        


def correct_init_whenreplace(init):
    init_correct_sign = np.sign(init)
    init_correct_sign[init == 0] = 1  
    return init_correct_sign

def get_argument_names(func):
    signature = inspect.signature(func)
    return [param.name for param in signature.parameters.values()]

def applyfunctiontothedictvalues(func, dict_, x_array):
    argument_names = get_argument_names(func)
    parameter_values = [dict_[key]["values"] for key in argument_names if key != "init"]
    array_value = np.array([func(x, *params) for x, *params in zip(x_array, *parameter_values)])
    return array_value

def vectorize_function(func):
    def func_vector(x, dict):
        number_iterations = len(next(iter(dict.values()))["values"])
        x_array = extend_init(x, number_iterations)
        array_values = applyfunctiontothedictvalues(func, dict, x_array)
        return array_values
    func_vector.__name__ = f"{func.__name__}_vector"
    return func_vector

def extend_init(x, number_iterations):
    if not isinstance(x[0], np.ndarray):
        x_array = [x] * number_iterations
    else:
        x_array = [x[0]] * number_iterations
    return np.array(x_array)


def check_positions(listtupples, positions):
    
    return [t in positions for t in listtupples]



def Merge(dict_1, dict_2):
	result = dict_1 | dict_2
	return result


####
def convert_cat_parameters_switch_act(list_input_cat):
    
    """  """
    
    
    uniq = np.unique(list_input_cat).tolist()

    dict_= {}
    for input_ in uniq:
        
        values=[(input_==i)*1 for i in list_input_cat]
        
        dict_[input_]={"values":values}
        
    return dict_ 






def generate_list_from_switch(values_p,lensample):
    
    """ For a categorical parameter associated with switch parameters, draws stochastic values for these switch parameters """
    
    if sum(values_p)!=1:
        print("ERROR sum of probabilities for switch not 1",sum(values_p))
        #sys.exit()
    
    num_rows = len(values_p) # number of categories for the switch parameter, ex wheat, soy, alfalfa
    num_cols = lensample # number of iterations
    
    array = np.zeros((num_rows, num_cols), dtype=int)
    
    # Determine the position of '1' in each column based on probabilities

    for col in range(num_cols):
        
        rand = np.random.random()
        cumulative_prob = 0
        
        for row in range(num_rows):
            
            cumulative_prob += values_p[row]
            if rand < cumulative_prob:
                array[row, col] = 1
                break
 
    # restructure      
    list_of_listvalues_switch=[]
    for row in range(num_rows):     
        
        list_of_listvalues_switch.append(array[row,].tolist())


    return list_of_listvalues_switch



def sampling_func_lhs(dictionaries, size):
    '''Function which returns a Latin Hypercube sample for the input space of 
    parameters in the dictionaries, assuming all parameters have uniform distributions.
    '''
    
    names_param = []
    bounds = []
    
    all_unique_param_and_values = []
    all_switch_param = []
    
    list_values_output_switch = []
    names_param_switch = []
    names_param_unique = []
    
    for key in dictionaries:
        dict_param = dictionaries[key]
    
        dict_param_input = dict_param.copy()
        
        unique_params = []
        switch_params = []
        
        for param in dict_param_input:
            if dict_param_input[param][0] == 'unique':
                names_param_unique.append(param)
                unique_params.append(param)
                all_unique_param_and_values.append((param, dict_param_input[param][1][0]))

            if dict_param_input[param][0] == 'switch':
                switch_params.append(param)
                all_switch_param.append(dict_param_input[param])
                                
        for a in unique_params:
            dict_param_input.pop(a)
        for a in switch_params:
            dict_param_input.pop(a)

        for param in dict_param_input:
            distrib = dict_param_input[param][0]
            if distrib == 'unif':
                bounds.append([dict_param_input[param][1][1], dict_param_input[param][1][2]])
                names_param.append(param)

    # Generate LHS samples for the entire parameter space
    lhs = LatinHypercube(d=len(names_param))
    lhs_samples = lhs.random(n=size)

    # Scale the LHS samples according to the specified uniform bounds
    sample_array = np.zeros((size, len(names_param)), dtype="float32")
    for i, bound in enumerate(bounds):
        lower, upper = bound
        sample_array[:, i] = lower + lhs_samples[:, i] * (upper - lower)

    sample_dataframe = pd.DataFrame(sample_array, columns=names_param)
    
    # Add parameters with fixed/unique values
    for param in all_unique_param_and_values:
        sample_dataframe[param[0]] = param[1]
    
    # Add the switch parameters
    for param in all_switch_param:
        values_p = param[2]
        names_options = param[3]
        
        list_values_output_switch = generate_list_from_switch(values_p, size)
        names_param_switch = [name + "_switch" for name in names_options]
        
        for option in range(len(names_param_switch)):
            sample_dataframe[names_param_switch[option]] = list_values_output_switch[option]

    names_param_total = names_param + names_param_unique + names_param_switch
    
    return names_param_total, sample_dataframe

# Make sure to define the generate_list_from_switch function or import it if it's defined elsewhere




#######
"""Functions parameterizing the LCA matrixes"""
#######



######## PV


# Mounting structure

  

# 0.000541666666666667 ; initial input value (m3) for concrete
def fconcrete_PV_mounting(init,
                           concrete_mount_upd_PV):
    """ Updates the amount of concrete for the mounting structure for conv PV"""
    
    new_value = init * concrete_mount_upd_PV  # init = original value in the matrix
    
    return new_value





def fconcrete_AVS_mounting(init,
                           concrete_mount_ratio_avs_pv,
                           concrete_mount_upd_PV
                           ):
    """ Updates the amount of concrete for the mounting structure in the AVS"""

    new_value = init * concrete_mount_upd_PV * concrete_mount_ratio_avs_pv
    
    return new_value


    
  


# aluminium

def falum_PV_mounting(init,
                           aluminium_mount_upd_PV):
    """ Updates the amount of aluminium for the mounting structure for conv PV"""

    
    new_value = init * aluminium_mount_upd_PV    
    
    return new_value





def falum_AVS_mounting(init,
                           aluminium_mount_ratio_avs_pv,
                           aluminium_mount_upd_PV
                           ):
    
    """ Updates the amount of aluminium for the mounting structure for AVS"""

    new_value = init * aluminium_mount_upd_PV * aluminium_mount_ratio_avs_pv
    
    return new_value





# Steel


def fsteel_PV_mounting(init,
                           steel_mount_upd_PV):
    
    """ Updates the amount of steel for the mounting structure for conv PV"""

    new_value = init * steel_mount_upd_PV   
    
    return new_value






def fsteel_AVS_mounting(init,
                           steel_mount_ratio_avs_pv,
                           steel_mount_upd_PV
                           ):
    
    """ Updates the amount of steel for the mounting structure for AVS"""
   
    new_value = init * steel_mount_upd_PV * steel_mount_ratio_avs_pv
    
    return new_value


# Corrugated board box



def fcorrbox_PV_mounting(init,
                           corrbox_mount_upd_PV):
    
    """ Updates the amount of corrugated box for the mounting structure for conv PV"""

    new_value = init * corrbox_mount_upd_PV   
    
    return new_value






def fcorrbox_AVS_mounting(init,
                           corrbox_mount_ratio_avs_pv,
                           corrbox_mount_upd_PV
                           ):
    """ Updates the amount of corrugated box for the mounting structure for AVS"""

    
    new_value = init * corrbox_mount_upd_PV * corrbox_mount_ratio_avs_pv
    
    return new_value


# polytehylene et polystirene



def fpoly_PV_mounting(init,
                           poly_mount_upd_PV):
    
    """ Updates the amount of plastic for the mounting structure for PV conv"""

    new_value = init * poly_mount_upd_PV 
    
    return new_value






def fpoly_AVS_mounting(init,
                           poly_mount_ratio_avs_pv,
                           poly_mount_upd_PV
                           ):
    
    """ Updates the amount of plastic for the mounting structure for AVS"""

    
    new_value = init * poly_mount_upd_PV * poly_mount_ratio_avs_pv
    
    return new_value




# zinc coat



def fzinc_PV_mounting(init,
                           zinc_mount_upd_PV):
    
    """ Updates the amount of zinc for the mounting structure for conv PV"""

    
    new_value = init * zinc_mount_upd_PV    

    return new_value






def fzinc_AVS_mounting(init,
                           zinc_mount_ratio_avs_pv,
                           zinc_mount_upd_PV
                           ):
    
    """ Updates the amount of zinc for the mounting structure for AVS"""

    new_value = init * zinc_mount_upd_PV * zinc_mount_ratio_avs_pv
    
    return new_value





def f_mountingpersystem_AVS(init,
         
                  surface_cover_fraction_AVS,
                  surface_cover_fraction_PV,
                  plant_lifetime_PV,
                  plant_lifetime_ratio_avs_pv):
    
    """Sets the amount of mounting structure for 1 ha of AVS"""

    
    new_value = correct_init_whenreplace(init)* 10000 *surface_cover_fraction_AVS/(plant_lifetime_PV*plant_lifetime_ratio_avs_pv)

    return new_value


def f_mountingpersystem_PV(init,
           
                  surface_cover_fraction_PV,
                  plant_lifetime_PV):
    
    """Sets the amount of mounting structure for 1 ha for conv PV"""

    
    new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_PV /plant_lifetime_PV

    return new_value






""" Electric installation"""


# iF AVS or PV not mentioned ,same for both



def f_electricinstallation_amount_perm2(init,
                                  mass_to_power_ratio_electric_instal_PV,
                                  panel_efficiency_PV):
    
    """Sets the amount of electric installation per m2 of panel"""
    
    new_value = correct_init_whenreplace(init)*mass_to_power_ratio_electric_instal_PV * panel_efficiency_PV
    
    # #print( mass_to_power_ratio_electric_instal_PV,panel_efficiency_PV,new_value )
    # #print("lEEEN", len(init))
    return new_value



"""Inverter"""

# (25171, 25172)

def f_input_inverter_m2_panel_AVS(init,
                              plant_lifetime_PV,
                              panel_efficiency_PV,
                              inverter_specific_weight_PV,
                              inverter_lifetime_PV,
                              plant_lifetime_ratio_avs_pv
                              ):
    
    """Sets the amount of inverter per m2 of panel for the AVS"""

    
    new_value = correct_init_whenreplace(init) *panel_efficiency_PV * inverter_specific_weight_PV * plant_lifetime_PV*plant_lifetime_ratio_avs_pv/inverter_lifetime_PV 
    
    return new_value


# (25171, 25173)
def f_input_inverter_m2_panel_PV(init,
                              plant_lifetime_PV,
                              panel_efficiency_PV,
                              inverter_specific_weight_PV,
                              inverter_lifetime_PV,
                              ):
    
    """Sets the amount of inverter per m2 of panel for the conv PV"""

    new_value = correct_init_whenreplace(init) *panel_efficiency_PV * inverter_specific_weight_PV * plant_lifetime_PV/inverter_lifetime_PV 
    
    return new_value





""" Panel """


def f_outputAVS_crop_main(init,
                          wheat_switch,
                          soy_switch,
                          alfalfa_switch,
                          crop_yield_ratio_avs_ref,
                          crop_yield_upd_ref,
                          init_yield_soy,
                          init_yield_wheat,
                          init_yield_alfalfa):
    
    """Sets the output of crop production for the AVS"""
    
    new_value = correct_init_whenreplace(init) * (init_yield_alfalfa*alfalfa_switch + init_yield_soy*soy_switch + init_yield_wheat*wheat_switch) * crop_yield_upd_ref*crop_yield_ratio_avs_ref

    return new_value



def f_outputAVS_elec_main(init,
                  panel_efficiency_PV,
                  surface_cover_fraction_PV,
                  surface_cover_fraction_AVS,
                  annual_producible_PV,
                  annual_producible_ratio_avs_pv):
    
    """Sets the output of electricity production for the AVS"""

    
    new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv


    #print("f_outputAVS_elec_main",new_value)

    return new_value



def f_outputAVS_elec_margi_crop_main(init,
                  panel_efficiency_PV,
                  surface_cover_fraction_PV,
                  surface_cover_fraction_AVS,
                  annual_producible_PV,
                  annual_producible_ratio_avs_pv,
                  substit_margi_switch):
    
    """Sets the output of electricity production for the AVS, when crop is the main product.
    For the substituion of marginal elec"""

    
    new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv
    
    
    new_value = new_value * substit_margi_switch
    
    ##print("f_outputAVS_elec_margi_crop_main",new_value)

    return new_value

def f_outputAVS_elec_PV_crop_main(init,
                  panel_efficiency_PV,
                  surface_cover_fraction_PV,
                  surface_cover_fraction_AVS,
                  annual_producible_PV,
                  annual_producible_ratio_avs_pv,
                  substit_PV_switch):
    
    """Sets the output of electricity production for the AVS, when crop is the main product.
    For the substitution of conv PV"""

    
    new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv 
    
    new_value = new_value * substit_PV_switch
    
    # nan_count = sum(math.isnan(x) for x in new_value)
    # loc_nan = [math.isnan(x) for x in new_value]
    # if nan_count!=0:
    #     #print("loc_nan",loc_nan)

    ##print("nan_count",nan_count)
    
    ##print("f_outputAVS_elec_PV_crop_main",init)

    return new_value
    




def f_outputPV(init,
                  panel_efficiency_PV,
                  surface_cover_fraction_PV,
                  annual_producible_PV):
    
    """Sets the output of electricity production for the conv PV"""

    
    new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_PV * panel_efficiency_PV* annual_producible_PV
    

    nan_count = sum(math.isnan(x) for x in new_value)
    if nan_count!=0:
        sys.exit("nan")


    return new_value


def f_panelperAVS(init,
                  surface_cover_fraction_AVS,
                  plant_lifetime_PV,
                  plant_lifetime_ratio_avs_pv):
    
    """Sets the amount of m2 of pV panels, per ha, for the AVS"""

    
    new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS/(plant_lifetime_PV*plant_lifetime_ratio_avs_pv)


    return new_value

def f_panelperPV(init,
                  surface_cover_fraction_PV,
                  plant_lifetime_PV):
    
    """Sets the amount of m2 of pV panels, per ha, for the conv PV"""

    new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_PV /plant_lifetime_PV
    ##print("new_value",new_value)
    # new_value =correct_init_whenreplace(init)
    
    # #print(init)
    # #print("new_value",new_value)
    return new_value




def f_aluminium_input_panel(init,
                            aluminium_panel_weight_PV):
    
    
    """Sets the amount of aluminium per panel frame"""

    new_value =  correct_init_whenreplace(init) *aluminium_panel_weight_PV 

    return new_value



"""wafer"""

def f_inputelec_wafer(init,
                      manufacturing_eff_wafer_upd_PV): #"""both on row and rer"""

    """Sets the amount of elec per m2 of wafer"""

    
    new_value = init * manufacturing_eff_wafer_upd_PV
    
    return new_value








""" Silicon production"""

 
def f_elec_intensity_solargraderow(init,
                                   solargrade_electric_intensity_PV):
    
    """ Updates the elec consumpotion for the silicon manufacture , for the row"""
    
    new_value = init*solargrade_electric_intensity_PV/110 # 110 kWh in total in the original act

    return new_value


def f_elec_intensity_solargraderer(init,
                                    solargrade_electric_intensity_PV):
    
    """ Updates the elec consumpotion for the silicon manufacture , for the rer"""

    
    new_value = init*solargrade_electric_intensity_PV/110
    
    return new_value






"""iLUC"""


def f_NPP_weighted_ha_y_landmarket_cropref(init,
       NPP_weighted_ha_y_eq_cropref):
    
    """ Sets the amount of NPP equivalent for 1 ha used by the conventional crop """
    
    new_value = correct_init_whenreplace(init) * NPP_weighted_ha_y_eq_cropref
    
    return new_value




def f_NPP_weighted_ha_y_landmarket_AVS(init,
       NPP_weighted_ha_y_eq_cropref,
       NPP_weighted_ha_y_eq_ratio_avs_cropref):
    
    """ Sets the amount of NPP equivalent for 1 ha used by the AVS """

    
    new_value = correct_init_whenreplace(init) * NPP_weighted_ha_y_eq_cropref * NPP_weighted_ha_y_eq_ratio_avs_cropref
    
    return new_value


def f_NPP_weighted_ha_y_landmarket_PV(init,
       NPP_weighted_ha_y_eq_cropref,
       NPP_weighted_ha_y_eq_ratio_pv_cropref):
    
    """ Sets the amount of NPP equivalent for 1 ha used by the PV """

    
    new_value = correct_init_whenreplace(init) * NPP_weighted_ha_y_eq_cropref * NPP_weighted_ha_y_eq_ratio_pv_cropref
    
    return new_value




def f_iluc_landmarket_PV(init,
       iluc_lm_PV):
    
    """ iluc GW impact for 1  NPP equivalent for the conv PV landmarket"""

    
    new_value = correct_init_whenreplace(init)*iluc_lm_PV
    
    return new_value


def f_iluc_landmarket_AVS(init,
       iluc_lm_AVS):
    
    """ iluc GW impact for 1 NPP equivalent for the AVS landmarket"""

    
    new_value = correct_init_whenreplace(init)*iluc_lm_AVS
    
    return new_value


def f_iluc_landmarket_cropref(init,
       iluc_lm_cropref):
    
    """ iluc GW impact for 1 NPP equivalent for the crop landmarket"""

    new_value = correct_init_whenreplace(init)*iluc_lm_cropref
    
    return new_value



""" Agri """

def f_switch_wheat(init,
                 wheat_switch):
    
    """ Assigns 0 or 1 to choose wheat as the modeled crop"""

    
    new_value = correct_init_whenreplace(init) * wheat_switch
    
    return new_value



def f_switch_soy(init,
                 soy_switch):
    
    """ Assigns 0 or 1 to choose soy as the modeled crop"""

    new_value = correct_init_whenreplace(init) * soy_switch
    
    return new_value

def f_switch_alfalfa(init,
                 alfalfa_switch):
    
    """ Assigns 0 or 1 to choose alfalfa as the modeled crop"""

    
    new_value = correct_init_whenreplace(init) * alfalfa_switch
    
    return new_value





def f_output_crop_ref(init,
                   crop_yield_upd_ref):
    
    """ Updates the output of the conv crop production"""

    
    new_value = init*crop_yield_upd_ref
    #print("f_output_crop_ref",new_value)
    
    return new_value


def f_output_crop_avs(init,
                   crop_yield_upd_ref,
                   crop_yield_ratio_avs_ref):
    
    """ Updates the output of the AVS crop production"""

    
    new_value = init*crop_yield_upd_ref *crop_yield_ratio_avs_ref
    
    #print("f_output_crop_avs",new_value)

    return new_value


def f_fert_input_ref(init,
                      crop_fert_upd_ref):
    
    """ Updates the amount of fertilizer input to conv crop production"""

    
    new_value = init *crop_fert_upd_ref
    
    return new_value




def f_fert_input_avs(init,
                       crop_fert_upd_ref,
                       crop_fert_ratio_avs_ref):
    
    """ Updates the amount of fertilizer input to AVS"""

    
    new_value =  init * crop_fert_upd_ref *crop_fert_ratio_avs_ref
    
    return new_value


def f_nutri_emission_ref(init,
                   crop_fert_upd_ref):
    
    """ Updates the amount of emissions associated with fertilizer for conv crop prod"""

    
    new_value = init *crop_fert_upd_ref
    
    return new_value



def f_nutri_emission_avs(init,
                   crop_fert_upd_ref,
                   crop_fert_ratio_avs_ref):
    
    """ Updates the amount of emissions associated with fertilizer for AVS"""

    
    new_value = init * crop_fert_upd_ref *crop_fert_ratio_avs_ref
    
    return new_value



def f_machinery_ref(init,
                   crop_mach_upd_ref):
    
    """ Updates the amount of machinery for conv crop prod"""

    
    new_value = init * crop_mach_upd_ref 
    
    return new_value



def f_machinery_avs(init,
                   crop_mach_upd_ref,
                   crop_mach_ratio_avs_ref):
    
    """ Updates the amount of machinery for AVS"""

    
    new_value = init * crop_mach_upd_ref * crop_mach_ratio_avs_ref
    
    return new_value



def f_water_ref(init,
                   water_upd_ref):
    
    """ Updates the amount of irrigated water for conv crop"""

    
    new_value = init * water_upd_ref
    
    return new_value


def f_water_avs(init,
                water_upd_ref,
                   water_ratio_avs_ref):
    
    """ Updates the amount of irrigated water for avs"""

    
    new_value = init * water_upd_ref * water_ratio_avs_ref
    
    return new_value



def f_carbon_soil_accumulation_ref(init,
                carbon_accumulation_soil_ref):
    
    """ Sets the amount of carbon accumulated for the conv crop"""

    
    new_value = correct_init_whenreplace(init) * carbon_accumulation_soil_ref
    
    return new_value


def f_carbon_soil_accumulation_avs(init,
                carbon_accumulation_soil_AVS):
    
    """ Sets the amount of carbon accumulated for AVS"""

    
    new_value = correct_init_whenreplace(init) * carbon_accumulation_soil_AVS
    
    return new_value


def f_carbon_soil_accumulation_pv(init,
                carbon_accumulation_soil_PV):
    
    """ Sets the amount of carbon accumulated for PV"""

    
    new_value = correct_init_whenreplace(init) * carbon_accumulation_soil_PV
    
    return new_value





# Multi marginal_elec

def modif_impact_marginal_elec(init,
                impact_update_margi):
    
    """  Updates the impact of  the marginal electricity"""

    
    new_value = init * (impact_update_margi-1)
    
    return new_value





def fdelete(init,
                   cover_av,
                   lifetime):
    
    """deletes a flow"""
    
    new_value = 0
    
    return new_value







""" Collects the functions in a dictionnary, together with the locations where they should be applied in the matrix """

dict_funct = { "f1": {"func":vectorize_function(fconcrete_PV_mounting), # returns a vectorized function applying the function on the vector of stochastic parameters values
                      "indices":[(concrete_mount.id,mount_system_PV.id),  #   # where it should apply on the matrix (row, col) : the position of the flow to modify
                                (concrete_mount_waste_1.id,mount_system_PV.id),
                                (concrete_mount_waste_2.id,mount_system_PV.id),
                                (concrete_mount_waste_3.id,mount_system_PV.id)]},    

                "f2": {"func":vectorize_function(fconcrete_AVS_mounting),     
                                    "indices":[(concrete_mount.id,mount_system_AVS.id),
                                                (concrete_mount_waste_1.id,mount_system_AVS.id),
                                                (concrete_mount_waste_2.id,mount_system_AVS.id),
                                                (concrete_mount_waste_3.id,mount_system_AVS.id)]},

                "f3": {"func":vectorize_function(falum_PV_mounting), # ok 17.03
                                      "indices":[(aluminium_mount.id,mount_system_PV.id),
                                                (alu_extru_mount.id,mount_system_PV.id),
                                                (alu_mount_scrap_1.id,mount_system_PV.id),
                                                (alu_mount_scrap_2.id,mount_system_PV.id),
                                                (alu_mount_scrap_3.id,mount_system_PV.id)]},
              

                "f4": {"func":vectorize_function(falum_AVS_mounting),   # ok 17.03
                                      "indices":[(aluminium_mount.id,mount_system_AVS.id),
                                                (alu_extru_mount.id,mount_system_AVS.id),
                                                (alu_mount_scrap_1.id,mount_system_AVS.id),
                                                (alu_mount_scrap_2.id,mount_system_AVS.id),
                                                (alu_mount_scrap_3.id,mount_system_AVS.id)]},
              
                "f5": {"func":vectorize_function(fsteel_PV_mounting),   # ok 17.03
                                      "indices":[(reinf_steel_mount.id,mount_system_PV.id),
                                                (chrom_steel_mount.id,mount_system_PV.id),
                                                (steel_rolling_mount.id,mount_system_PV.id),
                                                (wire_mount.id,mount_system_PV.id),
                                                (steel_mount_scrap_1.id,mount_system_PV.id),
                                                (steel_mount_scrap_2.id,mount_system_PV.id),
                                                (steel_mount_scrap_3.id,mount_system_PV.id)]},
              
                "f6": {"func":vectorize_function(fsteel_AVS_mounting),   # ok 17.03
                                      "indices":[(reinf_steel_mount.id,mount_system_AVS.id),
                                                (chrom_steel_mount.id,mount_system_AVS.id),
                                                (steel_rolling_mount.id,mount_system_AVS.id),
                                                (wire_mount.id,mount_system_AVS.id),
                                                (steel_mount_scrap_1.id,mount_system_AVS.id),
                                                (steel_mount_scrap_2.id,mount_system_AVS.id),
                                                (steel_mount_scrap_3.id,mount_system_AVS.id)]},
              
                "f7": {"func":vectorize_function(fcorrbox_PV_mounting),  # ok 17.03
                                      "indices":[(cor_box_mount_1.id,mount_system_PV.id),
                                                (cor_box_mount_2.id,mount_system_PV.id),
                                                (cor_box_mount_3.id,mount_system_PV.id),
                                                (cor_box_mount_4.id,mount_system_PV.id),
                                                (cor_box_mount_waste_1.id,mount_system_PV.id),
                                                (cor_box_mount_waste_2.id,mount_system_PV.id),
                                                (cor_box_mount_waste_3.id,mount_system_PV.id)]},
              
                "f8": {"func":vectorize_function(fcorrbox_AVS_mounting), # ok 17.03
                                      "indices":[(cor_box_mount_1.id,mount_system_AVS.id),
                                                (cor_box_mount_2.id,mount_system_AVS.id),
                                                (cor_box_mount_3.id,mount_system_AVS.id),
                                                (cor_box_mount_4.id,mount_system_AVS.id),
                                                (cor_box_mount_waste_1.id,mount_system_AVS.id),
                                                (cor_box_mount_waste_2.id,mount_system_AVS.id),
                                                (cor_box_mount_waste_3.id,mount_system_AVS.id)]},
              
                "f9": {"func":vectorize_function(fpoly_PV_mounting), # ok 17.03
                                      "indices":[(poly_mount_1.id,mount_system_PV.id),
                                                (poly_mount_2.id,mount_system_PV.id),
                                                (poly_mount_waste_1.id,mount_system_PV.id),
                                                (poly_mount_waste_2.id,mount_system_PV.id),
                                                (poly_mount_waste_3.id,mount_system_PV.id),
                                                (poly_mount_waste_4.id,mount_system_PV.id),
                                                (poly_mount_waste_5.id,mount_system_PV.id),
                                                (poly_mount_waste_6.id,mount_system_PV.id)]},              
              
                  "f10": {"func":vectorize_function(fpoly_AVS_mounting),  # ok 17.03
                                      "indices":[(poly_mount_1.id,mount_system_AVS.id),
                                                  (poly_mount_2.id,mount_system_AVS.id),
                                                  (poly_mount_waste_1.id,mount_system_AVS.id),
                                                  (poly_mount_waste_2.id,mount_system_AVS.id),
                                                  (poly_mount_waste_3.id,mount_system_AVS.id),
                                                  (poly_mount_waste_4.id,mount_system_AVS.id),
                                                  (poly_mount_waste_5.id,mount_system_AVS.id),
                                                  (poly_mount_waste_6.id,mount_system_AVS.id)]},
              
                  "f11": {"func":vectorize_function(fzinc_PV_mounting), # ok 17.03
                                      "indices":[(zinc_coat_mount_1.id,mount_system_PV.id),
                                                  (zinc_coat_mount_2.id,mount_system_PV.id)]},
              
                  "f12": {"func":vectorize_function(fzinc_AVS_mounting), # ok 17.03
                                      "indices":[(zinc_coat_mount_1.id,mount_system_AVS.id),
                                                  (zinc_coat_mount_2.id,mount_system_AVS.id)]},
              
              
              
               # Electric installation
              
              
                  "f13": {"func":vectorize_function(f_electricinstallation_amount_perm2),
                                       "indices":[(elecinstakg.id, pv_insta_AVS.id), # Into AVS 
                                                  (elecinstakg.id, pv_insta_PV.id)]}, # Into PV  # Ok 17.03
              

              
              
              # #   # AVS 
              
                    "f14": {"func":vectorize_function(f_outputAVS_elec_main),
                                         "indices":[(AVS_elec_main.id, AVS_elec_main.id)]},  # Ok  17.03
              
                   "f15": {"func":vectorize_function(f_outputPV),
                                        "indices":[(PV_ref.id, PV_ref.id)]}, # Ok 17.03
                   
                   "f16": {"func":vectorize_function(f_outputAVS_elec_margi_crop_main),
                                        "indices":[(elec_marginal_fr_copy.id, AVS_crop_main.id)]}, # Ok 17.03      
                   
                
                   "f17": {"func":vectorize_function(f_outputAVS_elec_PV_crop_main),
                                        "indices":[(PV_ref.id, AVS_crop_main.id)]}, # Ok 17.03      
                   
                
                
                

                
                
                
                
                
                
                
              
                  "f18": {"func":vectorize_function(f_panelperAVS),   # Ok 17.03
                                        "indices":[(pv_insta_AVS.id, AVS_elec_main.id),
                                                   (pv_insta_AVS.id, AVS_crop_main.id)]},
              

                  "f19": {"func":vectorize_function(f_panelperPV),
                                         "indices":[(pv_insta_PV.id, PV_ref.id)]}, # Ok 17.03
   
                  "f20": {"func":vectorize_function(f_mountingpersystem_AVS),
                                         "indices":[(mount_system_AVS.id, AVS_elec_main.id),
                                                    (mount_system_AVS.id, AVS_crop_main.id)]}, # Ok 17.03
              
                  "f21": {"func":vectorize_function(f_mountingpersystem_PV),
                                         "indices":[(mount_system_PV.id, PV_ref.id)]}, # Ok 17.03




                 #PV
              
              
                    "f22": {"func":vectorize_function(f_aluminium_input_panel), # Ok 17.03
                                         "indices":[(aluminium_panel.id, pvpanel_prod_row.id),
                                                    (aluminium_panel.id, pvpanel_prod_rer.id)]},
              
                   "f23": {"func":vectorize_function(f_inputelec_wafer), # Ok 17.03
                                        "indices":[(elec_wafer_nz.id, wafer_row.id),
                                                   (elec_wafer_rla.id, wafer_row.id),
                                                   (elec_wafer_raf.id, wafer_row.id),
                                                   (elec_wafer_au.id, wafer_row.id),
                                                   (elec_wafer_ci.id, wafer_row.id),
                                                   (elec_wafer_rna.id, wafer_row.id),
                                                   (elec_wafer_ras.id,wafer_row.id),
                                                   (elec_wafer_rer.id, wafer_rer.id)
                                              
                                                   ]},
              
              
              
               #  # Silicon
              
                    "f24": {"func":vectorize_function(f_elec_intensity_solargraderow),# Ok 17.03
                                         "indices":[(elec_sili_raf.id, si_sg_row.id),
                                                    (elec_sili_au.id, si_sg_row.id),
                                                    (elec_sili_ci.id, si_sg_row.id),
                                                    (elec_sili_nz.id, si_sg_row.id),
                                                    (elec_sili_ras.id, si_sg_row.id),
                                                    (elec_sili_rna.id, si_sg_row.id),
                                                    (elec_sili_rla.id, si_sg_row.id)
                                              
                                              
                                                    ]},    

                    "f25": {"func":vectorize_function(f_elec_intensity_solargraderer), # Ok 17.03
                                         "indices":[(electricity_RER_margetgroup.id, si_sg_rer.id)]},   



               #    # Inverter
                
                
                
                  "f26": {"func":vectorize_function(f_input_inverter_m2_panel_AVS),
                                       "indices":[(mark_inv_500kW_kg.id, pv_insta_AVS.id)]},    # ok 17.03

                
                  "f27": {"func":vectorize_function(f_input_inverter_m2_panel_PV),
                                       "indices":[(mark_inv_500kW_kg.id, pv_insta_PV.id)]},     # ok 17.03

                        
               #  # iLUC
              
                  "f28": {"func":vectorize_function(f_NPP_weighted_ha_y_landmarket_cropref),
                                       "indices":[(LUCmarket_cropref.id, wheat_fr_ref.id),
                                                  (LUCmarket_cropref.id, soy_ch_ref.id),
                                                  (LUCmarket_cropref.id, alfalfa_ch_ref.id)]},     # ok 17.03
                  #explore_act(wheat_fr_ref)
                
                
                
                   "f29": {"func":vectorize_function(f_NPP_weighted_ha_y_landmarket_AVS),
                                        "indices":[(LUCmarket_AVS.id, AVS_elec_main.id),
                                                   (LUCmarket_AVS.id, AVS_crop_main.id)]},     # ok 17.03

                   "f30": {"func":vectorize_function(f_NPP_weighted_ha_y_landmarket_PV),
                                        "indices":[(LUCmarket_PVref.id, PV_ref.id)]},     # ok 17.03

                   "f31": {"func":vectorize_function(f_iluc_landmarket_PV),
                                        "indices":[(iluc.id, LUCmarket_PVref.id)]},     # ok 17.03
             
                   "f32": {"func":vectorize_function(f_iluc_landmarket_AVS),
                                        "indices":[(iluc.id, LUCmarket_AVS.id)]}, 
              
                   "f33": {"func":vectorize_function(f_iluc_landmarket_cropref),
                                         "indices":[(iluc.id, LUCmarket_cropref.id)]},
              
               # # # Agri


                 "f34": {"func":vectorize_function(f_switch_wheat),
                                      "indices":[(wheat_fr_AVS_elec_main.id, AVS_elec_main.id),
                                                 (wheat_fr_AVS_crop_main.id, AVS_crop_main.id)]} ,             
              
                 "f35": {"func":vectorize_function(f_switch_soy),
                                     "indices":[(soy_ch_AVS_elec_main.id, AVS_elec_main.id),
                                                (soy_ch_AVS_crop_main.id, AVS_crop_main.id)]},   
                 
                 "f36": {"func":vectorize_function(f_switch_alfalfa),
                                      "indices":[(alfalfa_ch_AVS_elec_main.id, AVS_elec_main.id),
                                                 (alfalfa_ch_AVS_crop_main.id, AVS_crop_main.id)]},
 
                 "f37": {"func":vectorize_function(f_output_crop_ref),
                                      "indices":[(soy_ch_ref.id, soy_ch_ref.id),
                                                 (wheat_fr_ref.id, wheat_fr_ref.id),
                                                 (alfalfa_ch_ref.id, alfalfa_ch_ref.id)]},   
              

                 "f38": {"func":vectorize_function(f_output_crop_avs),
                                      "indices":[(wheat_fr_ref.id, wheat_fr_AVS_elec_main.id),
                                                 (soy_ch_ref.id, soy_ch_AVS_elec_main.id),
                                                 (alfalfa_ch_ref.id, alfalfa_ch_AVS_elec_main.id)]},   # Because the output of this virtual activity is always 1 virtual unit. Here we modify the "actual" output which is the crop substitutiog the "ref" activity                                          
             
                 "f39": {"func":vectorize_function(f_fert_input_ref),
                                      "indices":[(ammonium_nitrate.id, wheat_fr_ref.id),
                                                 (ammonium_sulfate.id, wheat_fr_ref.id),
                                                 (urea.id, wheat_fr_ref.id),
                                                 (fert_broadcaster.id, wheat_fr_ref.id),
                                                 (ino_P205_fr.id, wheat_fr_ref.id),
                                                 (org_P205.id, wheat_fr_ref.id),
                                                 (packaging_fert_glo.id, wheat_fr_ref.id),
                                                 (carbondioxide_fossil_urea.id, wheat_fr_ref.id),
                                              
                                              
                                                 (fert_broadcaster_ch.id, soy_ch_ref.id),
                                                 (green_manure_ch.id, soy_ch_ref.id),
                                                 (nutrient_supply_thomas_meal_ch.id, soy_ch_ref.id),
                                                 (liquidmanure_spr_ch.id, soy_ch_ref.id),
                                                 (packaging_fert_glo.id, soy_ch_ref.id),
                                                 (phosphate_rock_glo.id, soy_ch_ref.id),
                                                 (potassium_chloride_rer.id, soy_ch_ref.id),
                                                 (potassium_sulfate_rer.id, soy_ch_ref.id),
                                                 (single_superphosphate_rer.id, soy_ch_ref.id),
                                                 (solidmanure_spreading_ch.id, soy_ch_ref.id),
                                                 (triplesuperphosphate.id, soy_ch_ref.id),

                                                 (fert_broadcaster_ch.id, alfalfa_ch_ref.id),
                                                 (ino_P205_ch.id, alfalfa_ch_ref.id),
                                                 (liquidmanure_spr_ch.id, alfalfa_ch_ref.id),
                                                 (packaging_fert_glo.id, alfalfa_ch_ref.id),
                                                 (solidmanure_spreading_ch.id, alfalfa_ch_ref.id)]},   # Because the output of this virtual activity is always 1 virtual unit. Here we modify the "actual" output which is the crop substitutiog the "ref" activity                                          
                           
                 "f40": {"func":vectorize_function(f_fert_input_avs),
                                      "indices":[
                                                  # Elec main
                                          (ammonium_nitrate.id, wheat_fr_AVS_elec_main.id),
                                                 (ammonium_sulfate.id, wheat_fr_AVS_elec_main.id),
                                                 (urea.id, wheat_fr_AVS_elec_main.id),
                                                 (fert_broadcaster.id, wheat_fr_AVS_elec_main.id),
                                                 (ino_P205_fr.id, wheat_fr_AVS_elec_main.id),
                                                 (org_P205.id, wheat_fr_AVS_elec_main.id),
                                                 (packaging_fert_glo.id, wheat_fr_AVS_elec_main.id),
                                                 (carbondioxide_fossil_urea.id, wheat_fr_AVS_elec_main.id),
                                              
                                              
                                                 (fert_broadcaster_ch.id, soy_ch_AVS_elec_main.id),
                                                 (green_manure_ch.id, soy_ch_AVS_elec_main.id),
                                                 (nutrient_supply_thomas_meal_ch.id, soy_ch_AVS_elec_main.id),
                                                 (liquidmanure_spr_ch.id, soy_ch_AVS_elec_main.id),
                                                 (packaging_fert_glo.id, soy_ch_AVS_elec_main.id),
                                                 (phosphate_rock_glo.id, soy_ch_AVS_elec_main.id),
                                                 (potassium_chloride_rer.id, soy_ch_AVS_elec_main.id),
                                                 (potassium_sulfate_rer.id, soy_ch_AVS_elec_main.id),
                                                 (single_superphosphate_rer.id, soy_ch_AVS_elec_main.id),
                                                 (solidmanure_spreading_ch.id, soy_ch_AVS_elec_main.id),
                                                 (triplesuperphosphate.id, soy_ch_AVS_elec_main.id),

                                                 (fert_broadcaster_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                 (ino_P205_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                 (liquidmanure_spr_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                 (packaging_fert_glo.id, alfalfa_ch_AVS_elec_main.id),
                                                 (solidmanure_spreading_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                 
                                                 # Crop main
                                                 
                                                 (ammonium_nitrate.id, wheat_fr_AVS_crop_main.id),
                                                            (ammonium_sulfate.id, wheat_fr_AVS_crop_main.id),
                                                            (urea.id, wheat_fr_AVS_crop_main.id),
                                                            (fert_broadcaster.id, wheat_fr_AVS_crop_main.id),
                                                            (ino_P205_fr.id, wheat_fr_AVS_crop_main.id),
                                                            (org_P205.id, wheat_fr_AVS_crop_main.id),
                                                            (packaging_fert_glo.id, wheat_fr_AVS_crop_main.id),
                                                            (carbondioxide_fossil_urea.id, wheat_fr_AVS_crop_main.id),
                                                         
                                                         
                                                            (fert_broadcaster_ch.id, soy_ch_AVS_crop_main.id),
                                                            (green_manure_ch.id, soy_ch_AVS_crop_main.id),
                                                            (nutrient_supply_thomas_meal_ch.id, soy_ch_AVS_crop_main.id),
                                                            (liquidmanure_spr_ch.id, soy_ch_AVS_crop_main.id),
                                                            (packaging_fert_glo.id, soy_ch_AVS_crop_main.id),
                                                            (phosphate_rock_glo.id, soy_ch_AVS_crop_main.id),
                                                            (potassium_chloride_rer.id, soy_ch_AVS_crop_main.id),
                                                            (potassium_sulfate_rer.id, soy_ch_AVS_crop_main.id),
                                                            (single_superphosphate_rer.id, soy_ch_AVS_crop_main.id),
                                                            (solidmanure_spreading_ch.id, soy_ch_AVS_crop_main.id),
                                                            (triplesuperphosphate.id, soy_ch_AVS_crop_main.id),

                                                            (fert_broadcaster_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                            (ino_P205_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                            (liquidmanure_spr_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                            (packaging_fert_glo.id, alfalfa_ch_AVS_crop_main.id),
                                                            (solidmanure_spreading_ch.id, alfalfa_ch_AVS_crop_main.id)]},   # Because the output of this virtual activity is always 1 virtual unit. Here we modify the "actual" output which is the crop substitutiog the "AVS" activity                                          
                           
                  "f41": {"func":vectorize_function(f_nutri_emission_ref),
                                        "indices":[(ammonia.id, wheat_fr_ref.id),   #wheat_fr_ref.id
                                                  (dinitrogen_monoxide.id, wheat_fr_ref.id),
                                                  (nitrogen_oxide.id, wheat_fr_ref.id),
                                                  (nitrate.id, wheat_fr_ref.id),
                                                  (phosphate_groundwater.id, wheat_fr_ref.id),
                                                  (phosphate_surfacewater.id, wheat_fr_ref.id),
                                              
                                                  (ammonia.id, soy_ch_ref.id),
                                                    (dinitrogen_monoxide.id, soy_ch_ref.id),
                                                    (nitrogen_oxide.id, soy_ch_ref.id),
                                                    (nitrate.id, soy_ch_ref.id),
                                                    (phosphate_groundwater.id, soy_ch_ref.id),
                                                    (phosphate_surfacewater.id, soy_ch_ref.id),
                                              
                                                  (ammonia.id, alfalfa_ch_ref.id),
                                                    (dinitrogen_monoxide.id, alfalfa_ch_ref.id),
                                                    (nitrogen_oxide.id, alfalfa_ch_ref.id),
                                                    (nitrate.id, alfalfa_ch_ref.id),
                                                    (phosphate_groundwater.id, alfalfa_ch_ref.id),
                                                    (phosphate_surfacewater.id, alfalfa_ch_ref.id)
                                                      ]},
              
                  "f42": {"func":vectorize_function(f_nutri_emission_avs),
                                        "indices":[
                                                # Elec main
                                                (ammonia.id, wheat_fr_AVS_elec_main.id),
                                                  (dinitrogen_monoxide.id, wheat_fr_AVS_elec_main.id),
                                                  (nitrogen_oxide.id, wheat_fr_AVS_elec_main.id),
                                                  (nitrate.id, wheat_fr_AVS_elec_main.id),
                                                  (phosphate_groundwater.id, wheat_fr_AVS_elec_main.id),
                                                  (phosphate_surfacewater.id, wheat_fr_AVS_elec_main.id),
                                              
                                                  (ammonia.id, soy_ch_AVS_elec_main.id),
                                                    (dinitrogen_monoxide.id, soy_ch_AVS_elec_main.id),
                                                    (nitrogen_oxide.id, soy_ch_AVS_elec_main.id),
                                                    (nitrate.id, soy_ch_AVS_elec_main.id),
                                                    (phosphate_groundwater.id, soy_ch_AVS_elec_main.id),
                                                    (phosphate_surfacewater.id, soy_ch_AVS_elec_main.id),
                                              
                                                  (ammonia.id, alfalfa_ch_AVS_elec_main.id),
                                                    (dinitrogen_monoxide.id, alfalfa_ch_AVS_elec_main.id),
                                                    (nitrogen_oxide.id, alfalfa_ch_AVS_elec_main.id),
                                                    (nitrate.id, alfalfa_ch_AVS_elec_main.id),
                                                    (phosphate_groundwater.id, alfalfa_ch_AVS_elec_main.id),
                                                    (phosphate_surfacewater.id, alfalfa_ch_AVS_elec_main.id),
                                                    
                                                    # Crop main
                                                (ammonia.id, wheat_fr_AVS_crop_main.id),
                                                  (dinitrogen_monoxide.id, wheat_fr_AVS_crop_main.id),
                                                  (nitrogen_oxide.id, wheat_fr_AVS_crop_main.id),
                                                  (nitrate.id, wheat_fr_AVS_crop_main.id),
                                                  (phosphate_groundwater.id, wheat_fr_AVS_crop_main.id),
                                                  (phosphate_surfacewater.id, wheat_fr_AVS_crop_main.id),
                                              
                                                  (ammonia.id, soy_ch_AVS_crop_main.id),
                                                    (dinitrogen_monoxide.id, soy_ch_AVS_crop_main.id),
                                                    (nitrogen_oxide.id, soy_ch_AVS_crop_main.id),
                                                    (nitrate.id, soy_ch_AVS_crop_main.id),
                                                    (phosphate_groundwater.id, soy_ch_AVS_crop_main.id),
                                                    (phosphate_surfacewater.id, soy_ch_AVS_crop_main.id),
                                              
                                                  (ammonia.id, alfalfa_ch_AVS_crop_main.id),
                                                    (dinitrogen_monoxide.id, alfalfa_ch_AVS_crop_main.id),
                                                    (nitrogen_oxide.id, alfalfa_ch_AVS_crop_main.id),
                                                    (nitrate.id, alfalfa_ch_AVS_crop_main.id),
                                                    (phosphate_groundwater.id, alfalfa_ch_AVS_crop_main.id),
                                                    (phosphate_surfacewater.id, alfalfa_ch_AVS_crop_main.id)
                                                    
                                                    
                                                    ]},
              
              
                  "f43": {"func":vectorize_function(f_machinery_ref),
                                       "indices":[(tillage_rotary_harrow_glo.id, wheat_fr_ref.id),
                                                  (tillage_rotary_spring_tine_glo.id, wheat_fr_ref.id),
                                                  (sowing_glo.id, wheat_fr_ref.id),
                                                  (tillage_ploughing_glo.id, wheat_fr_ref.id),
                                              
                                                  (tillage_currying_weeder_ch.id, soy_ch_ref.id),
                                                    (tillage_rotary_spring_tine_ch.id, soy_ch_ref.id),
                                                    (sowing_ch.id, soy_ch_ref.id),
                                              
                                                  (fodder_loading_ch.id, alfalfa_ch_ref.id),
                                                    (rotary_mower_ch.id, alfalfa_ch_ref.id),
                                                    (sowing_ch.id, alfalfa_ch_ref.id),
                                                    (tillage_rotary_spring_tine_ch.id, alfalfa_ch_ref.id),
                                                    (tillage_ploughing_ch.id, alfalfa_ch_ref.id),
                                                    (tillage_rolling_ch.id, alfalfa_ch_ref.id)]},
              
                  "f44": {"func":vectorize_function(f_machinery_avs),
                                       "indices":[
                                           # elec main
                                           (tillage_rotary_harrow_glo.id, wheat_fr_AVS_elec_main.id),
                                                  (tillage_rotary_spring_tine_glo.id, wheat_fr_AVS_elec_main.id),
                                                  (sowing_glo.id, wheat_fr_AVS_elec_main.id),
                                                  (tillage_ploughing_glo.id, wheat_fr_AVS_elec_main.id),
                                              
                                                  (tillage_currying_weeder_ch.id, soy_ch_AVS_elec_main.id),
                                                    (tillage_rotary_spring_tine_ch.id, soy_ch_AVS_elec_main.id),
                                                    (sowing_ch.id, soy_ch_AVS_elec_main.id),
                                              
                                                  (fodder_loading_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                    (rotary_mower_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                    (sowing_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                    (tillage_rotary_spring_tine_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                    (tillage_ploughing_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                    (tillage_rolling_ch.id, alfalfa_ch_AVS_elec_main.id),
                                            # crop main
                                            
                                            
                                           (tillage_rotary_harrow_glo.id, wheat_fr_AVS_crop_main.id),
                                                  (tillage_rotary_spring_tine_glo.id, wheat_fr_AVS_crop_main.id),
                                                  (sowing_glo.id, wheat_fr_AVS_crop_main.id),
                                                  (tillage_ploughing_glo.id, wheat_fr_AVS_crop_main.id),
                                              
                                                  (tillage_currying_weeder_ch.id, soy_ch_AVS_crop_main.id),
                                                    (tillage_rotary_spring_tine_ch.id, soy_ch_AVS_crop_main.id),
                                                    (sowing_ch.id, soy_ch_AVS_crop_main.id),
                                              
                                                  (fodder_loading_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                    (rotary_mower_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                    (sowing_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                    (tillage_rotary_spring_tine_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                    (tillage_ploughing_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                    (tillage_rolling_ch.id, alfalfa_ch_AVS_crop_main.id)
                                                    ]},
              
                 "f45": {"func":vectorize_function(f_water_ref),
                                      "indices":[(water_air.id, wheat_fr_ref.id),
                                                 (water_ground.id, wheat_fr_ref.id),
                                                 (water_surface.id, wheat_fr_ref.id)]},
              
              
                 "f46": {"func":vectorize_function(f_water_avs),
                                      "indices":[(water_air.id, wheat_fr_AVS_elec_main.id),
                                                 (water_ground.id, wheat_fr_AVS_elec_main.id),
                                                 (water_surface.id, wheat_fr_AVS_elec_main.id),
                                                 
                                                 
                                                 (water_air.id, wheat_fr_AVS_crop_main.id),
                                                (water_ground.id, wheat_fr_AVS_crop_main.id),
                                                (water_surface.id, wheat_fr_AVS_crop_main.id)]},
              
                "f47": {"func":vectorize_function(f_carbon_soil_accumulation_ref),
                                     "indices":[(c_soil_accu.id, wheat_fr_ref.id),
                                                (c_soil_accu.id, soy_ch_ref.id),
                                                (c_soil_accu.id, alfalfa_ch_ref.id)]},
              
                "f48": {"func":vectorize_function(f_carbon_soil_accumulation_avs),
                                     "indices":[(c_soil_accu.id, AVS_elec_main.id),
                                                (c_soil_accu.id, AVS_crop_main.id)]},
   
                "f49": {"func":vectorize_function(f_carbon_soil_accumulation_pv),
                                     "indices":[(c_soil_accu.id, PV_ref.id)]},
                
                "f50": {"func":vectorize_function(f_outputAVS_crop_main),
                                     "indices":[(AVS_crop_main.id, AVS_crop_main.id)]}
                

              }












# Static dp fix : additional array for quick static fixes in the matric
 

# We delete the hydro elec for rer # cf Besseau
indices_array_fix = np.array([(electricity_hydro.id, si_sg_rer.id)],  # (electricity_hydro.id,siliconproductionsolar_grade_rer.id)
                             
                             dtype=bwp.INDICES_DTYPE)

data_array_fix = np.array(
    [0])




# Save datapackage Static dp fix

dp_static_fix = bwp.create_datapackage()



dp_static_fix.add_persistent_vector(
    matrix='technosphere_matrix',
    indices_array=indices_array_fix,
    data_array=data_array_fix,
    name='techno static fix')





