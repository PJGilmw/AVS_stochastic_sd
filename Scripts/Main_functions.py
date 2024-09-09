# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:45:17 2024

@author: pjouannais
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:54:48 2024

@author: pierre.jouannais


Script containing the main functions to compute the LCAS and use PRIM over the stochastic samples to return boxes.

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
import Parameters_and_functions
from Parameters_and_functions  import *

from util_functions import *


import pickle


import math 
from ema_workbench.analysis import prim

# from ema_workbench import ema_logging

import datetime
import os

import json
import ray


import logging

# Avoid plotting warning messages

logger = logging.getLogger('tune.logger')
logger.propagate = False










# ok 
def create_modif_arrays_para(act,
                              meth1,
                              name_db_foreground,
                              name_db_background,
                              name_db_additional_bio,
                              name_db_additional_bio_multi_margi,
                              list_totalindices,
                              dict_funct,
                              values_for_datapackages,
                              numberchunksforpara):
    
    """ Collects the original datapackages from ecoinvent and the foreground database and returns modified datapackages with computed stochastic values in the right positions."""
    
    fu, data_objs, _ = bd.prepare_lca_inputs(
        {act: 1}, method=meth1)  # Here are the original data objects
    
    
    
    # There should not be any duplicates, i.e, any flow which is affected by two functions.
    duplicates = set(x for x in list_totalindices if list_totalindices.count(x) > 1) 
    
    if len(duplicates)!=0:
        print("WARNING DUPLICATES", "\n",
              duplicates)
        #exit()
    
    
    

    
    # Due to parallel brightway import, the biosphere and the technosphere are not always in the same positions in the data_objs
    # little trick to make sure it always works:
        
    
    for data in data_objs:
        print(data.metadata["name"])
       # print(data.metadata)
        if data.metadata["name"]==name_db_background:
            
            # Technosphere background
    
            array_data_background, dict_ = data.get_resource(
                name_db_background+'_technosphere_matrix.data')
            array_flip_background, dict_ = data.get_resource(
                name_db_background+'_technosphere_matrix.flip')
            array_indice_background, dict_ = data.get_resource(
                name_db_background+'_technosphere_matrix.indices')
    
    
            # Biosphere
            array_data_biospherefromecoinvent, dict_ = data.get_resource(
                name_db_background+'_biosphere_matrix.data')
            array_indice_biospherefromecoinvent, dict_ = data.get_resource(
                name_db_background+'_biosphere_matrix.indices')
    
            
        elif data.metadata["name"]==name_db_foreground:
            

            array_data_foreground, dict_ = data.get_resource(
                name_db_foreground+'_technosphere_matrix.data')
            array_indice_foreground, dict_ = data.get_resource(
                name_db_foreground+'_technosphere_matrix.indices')
            array_flip_foreground, dict_ = data.get_resource(
                name_db_foreground+'_technosphere_matrix.flip')
    
            array_data_foreground_bio, dict_ = data.get_resource(
                name_db_foreground+'_biosphere_matrix.data')
            array_indice_foreground_bio, dict_ = data.get_resource(
                name_db_foreground+'_biosphere_matrix.indices')        
    
        elif data.metadata["name"]==name_db_additional_bio:
            

            array_data_additional_bio, dict_ = data.get_resource(
                name_db_additional_bio+'_technosphere_matrix.data')
            array_indice_additional_bio, dict_ = data.get_resource(
                name_db_additional_bio+'_technosphere_matrix.indices')
            

    
        elif data.metadata["name"]==name_db_additional_bio_multi_margi:
            

            array_data_additional_bio_multi_margi, dict_ = data.get_resource(
                name_db_additional_bio_multi_margi+'_technosphere_matrix.data')
            array_indice_additional_bio_multi_margi, dict_ = data.get_resource(
                name_db_additional_bio_multi_margi+'_technosphere_matrix.indices')
            

    
    # list_totalindices calculated in "parameters and functions script"
    
    # indices_strucutures contains all the tuples of indices that will require some modifications.
    
    # They can be in the ecoinvent tehcnosphere, the biosphere or the foreground 
    
    indices_structured = np.array(
        list_totalindices, dtype=array_indice_background.dtype)
    
    
    mask_foreground = np.isin(array_indice_foreground, indices_structured)
    #print(sum(mask_foreground))
    
    
    mask_background = np.isin(array_indice_background, indices_structured)
    #cc=sum(mask_background)
    #print(sum(mask_background))

    
    mask_biosphere_background = np.isin(array_indice_biospherefromecoinvent, indices_structured)
    #ii=sum(mask_biosphere_background)
    #print(sum(mask_biosphere_background))

    
    mask_biosphere_foreground = np.isin(array_indice_foreground_bio, indices_structured)
    #ee=sum(mask_biosphere_foreground)
    #print(sum(mask_biosphere_foreground))

    
    mask_additional_biosphere = np.isin(array_indice_additional_bio, indices_structured)
    #oo=sum(mask_additional_biosphere)
    #print(sum(mask_additional_biosphere))

    
    
    mask_additional_biosphere_multi_margi = np.isin(array_indice_additional_bio_multi_margi, indices_structured)
    #vv=sum(mask_additional_biosphere_multi_margi)
    #print(sum(mask_additional_biosphere_multi_margi))

    
    
    # These arrays contain all the data, indices and flip that will be modified by the functions, either in the foreground or the background
    
    # Technosphere
    
    data_array_total_init_techno = np.concatenate(
        (array_data_foreground[mask_foreground], 
         array_data_background[mask_background],
         array_data_additional_bio[mask_additional_biosphere],
         array_data_additional_bio_multi_margi[mask_additional_biosphere_multi_margi]
    ), axis=0)
    
    
    
    indices_array_total_init_techno = np.concatenate(
        (array_indice_foreground[mask_foreground],
         array_indice_background[mask_background],
         array_indice_additional_bio[mask_additional_biosphere],
         array_indice_additional_bio_multi_margi[mask_additional_biosphere_multi_margi]), axis=0)
    
    flip_array_total_init_techno = np.concatenate(
        (array_flip_foreground[mask_foreground],
         array_flip_background[mask_background],

         
         ), axis=0)
    
    # Biosphere
    
    
    data_array_total_init_bio = np.concatenate(
        (array_data_foreground_bio[mask_biosphere_foreground],
         array_data_biospherefromecoinvent[mask_biosphere_background]
    ), axis=0)
    
    
    
    indices_array_total_init_bio = np.concatenate(
        (array_indice_foreground_bio[mask_biosphere_foreground],
         array_indice_biospherefromecoinvent[mask_biosphere_background]), axis=0)
    
    
    
    
    
    # Divide the dictionnary of stochastic values for the parameters into chunks to be processed in parallel
    
    
    print("RRR")
    listof_values_for_datapackages, list_chunk_sizes = divide_dict(values_for_datapackages, numberchunksforpara)

    print("UUU")

    
    """ Create the modified data arrays """
    
    
    list_arrays_for_datapackages = []
    
    for values_for_datapackages_chunk in listof_values_for_datapackages:
        
        data_array_total_modif_techno = data_array_total_init_techno
        data_array_total_modif_bio = data_array_total_init_bio
        
        
        # going through the functions in dict_func and apply them where needed
        
        list_indices=[]
        for a in dict_funct:
            #print(a)
    
            
            # build a mask to know here the function applies
            mask_techno = check_positions(indices_array_total_init_techno.tolist(), dict_funct[a]["indices"])
            mask_bio = check_positions(indices_array_total_init_bio.tolist(), dict_funct[a]["indices"])
            #print("mask")

            if sum(mask_bio)==0 and sum(mask_techno)==0: # problem
                print("NOWHERE")
            
            list_indices.append(dict_funct[a]["indices"])
            
            if len(data_array_total_modif_techno)!=0: # if there are some modifications to apply on the technosphere
                
                # the data array is modified by applying the function (vectorized) over the stochasitic inputs in "values_for_datapackages_chunk"
                # Applied whee it should thangs to "map" and the mask. 
                # The function automatically picks the necessary input parameters in the dictionnary "values_for_datapackages_chunk" according to their names.
                
                data_array_total_modif_techno = np.where(mask_techno, dict_funct[a]["func"](
                    data_array_total_modif_techno, values_for_datapackages_chunk), data_array_total_modif_techno)
            
            if len(data_array_total_modif_bio)!=0:  # if there are some modifications to apply on the biosphere
        
                data_array_total_modif_bio = np.where(mask_bio, dict_funct[a]["func"](
                    data_array_total_modif_bio, values_for_datapackages_chunk), data_array_total_modif_bio)
            #print("apply")

        
    
        
        list_arrays_for_datapackages.append([data_array_total_modif_bio,
                             data_array_total_modif_techno,
                             indices_array_total_init_techno,
                             flip_array_total_init_techno,
                             indices_array_total_init_bio])



    return list_arrays_for_datapackages,values_for_datapackages,list_chunk_sizes













def create_dp_modif(data_array_total_modif_bio,
                     data_array_total_modif_techno,
                     indices_array_total_init_techno,
                     flip_array_total_init_techno,
                     indices_array_total_init_bio) : 
    """Returns a datapackage with the input arrays"""

   
    dp_modif = bwp.create_datapackage(sequential=True)
    
    # Transpose for the right structure
    data_array_total_modif_bio_T = data_array_total_modif_bio.T
    data_array_total_modif_techno_T = data_array_total_modif_techno.T
    
    
    if len(data_array_total_modif_techno_T)!=0:
        dp_modif.add_persistent_array(
            matrix='technosphere_matrix',
            indices_array=indices_array_total_init_techno,
            data_array=data_array_total_modif_techno_T,
            flip_array=flip_array_total_init_techno,
            name='techno modif')
    
    if len(data_array_total_modif_bio_T)!=0:
    
        dp_modif.add_persistent_array(
            matrix='biosphere_matrix',
            indices_array=indices_array_total_init_bio,
            data_array=data_array_total_modif_bio_T,
            name='bio modif')

    return dp_modif


def create_dp_static(indices_array_fix,
                     data_array_fix):
    """ Returns the static datapackage"""
    
    
    dp_static_fix = bwp.create_datapackage()



    dp_static_fix.add_persistent_vector(
        matrix='technosphere_matrix',
        indices_array=indices_array_fix,
        data_array=data_array_fix,
        name='techno static fix')

    return dp_static_fix
    
    
    
    
@ray.remote
def compute_stochastic_lcas_1worker(constant_inputs,
                            array_for_dp, chunk_size):
    
    """Computes stochatic lcas by calling the new parameterized, stochastic datapackages. 
    We need to load the bw objects in the function."""
    
    
    # UNPACK
    
    #print("UU")
    [list_fu,
    list_meth,
    uncert,
    C_matrixes,
    act,
    meth1,
    indices_array_fix,
    data_array_fix] = constant_inputs
    
    #print("OO")

    [data_array_total_modif_bio,
    data_array_total_modif_techno,
    indices_array_total_init_techno,
    flip_array_total_init_techno,
    indices_array_total_init_bio] = array_for_dp
    
    
    """ Load project, db, methods etc."""
    
    bd.projects.set_current('AVS')
    
    
    Ecoinvent = bd.Database('ecoinvent-3.10-consequential')
    
    biosphere = bd.Database('ecoinvent-3.10-biosphere')
    
    
    foregroundAVS = bd.Database("foregroundAVS")

    #print("ZZ")


    fu, data_objs, _ = bd.prepare_lca_inputs(
        {act: 1}, method=meth1)  # Here are the original data objects
        
    #print("PP")

    # CREATE DATAPACKAGE FROM MATRIXES ARRAYS


    dp_modif = create_dp_modif(data_array_total_modif_bio,
                         data_array_total_modif_techno,
                         indices_array_total_init_techno,
                         flip_array_total_init_techno,
                         indices_array_total_init_bio)
    
    
    dp_static_fix = create_dp_static(indices_array_fix,
                         data_array_fix)

    # With dp static fix
    # lca = bc.LCA(fu, data_objs = data_objs + [dp_static_fix] + [dp_modif],  use_arrays=True,use_distributions=uncert)
    # lca.lci()
    # lca.lcia()
    
    # without for test
    
    lca = bc.LCA(fu, data_objs = data_objs + [dp_static_fix] + [dp_modif],  use_arrays=True,use_distributions=uncert)
    lca.lci()
    lca.lcia()
    

    
    time1=time.time()  
    
    
    # Initalize the list of results 
    
    listarray_mc_sample = [np.array([[0]*len(list_fu)]*chunk_size,dtype="float32") for meth in range(len(list_meth))]
    
    
    for it in range(chunk_size):
        
        next(lca)
    
    
        for i in range(0,len(list_fu)):
            demand=list_fu[i][0]
            lca.redo_lcia({demand:1})  # redo with new FU
            
            index_array_method=-1
    
            for m in list_meth:
            
                #print("ok3",m)
    
                index_array_method+=1
            
                
                listarray_mc_sample[index_array_method][it,i]=(C_matrixes[m]*lca.inventory).sum() # This calculates the LCIA
                
    
    time2=time.time()
                
    tot_time = time2-time1 
    #print(tot_time)
    
    list_tables = [pd.DataFrame(listarray_mc_sample[i],columns=[fu_[1] for fu_ in list_fu]) for i in range(len(list_meth))]
    
    
    
    # The last row actually corresponds to the first one
    
    for table_index in range(len(list_tables)):
        #  add the last row as first row
        list_tables[table_index] = pd.concat([pd.DataFrame(list_tables[table_index].iloc[-1]).T, list_tables[table_index]], ignore_index=True)
    
        #  delete the last row 
        list_tables[table_index] = list_tables[table_index].drop(list_tables[table_index].index[-1])
    
    return list_tables


    
    
def compute_stochastic_lcas_para(
                            list_arrays_for_datapackages,
                            list_fu,
                            act,
                            list_meth,
                            uncert,
                            list_chunk_sizes,
                            indices_array_fix,
                            data_array_fix):
    """Computes stochastic LCAs by calling the new parameterized, stochastic datapackages"""
    
    # Characterization matrices necessary for computations
    Lca = bc.LCA({act: 1}, list_meth[0])
    Lca.lci()
    Lca.lcia()
    C_matrixes = load_characterization_matrix(list_meth, Lca)

    # Determine the number of chunks based on the number of available CPUs


    # Start parallel Monte Carlo simulation
    start_time = time.time()
  
    

    #print("XXXXXXX")
    
    ray.shutdown()

    
    
    
    # Get the absolute path of the current working directory
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)


    ray.init(runtime_env={"working_dir": current_directory},configure_logging=False, log_to_driver=False)
    
    constant_inputs = ray.put([list_fu,
                                list_meth,
                                uncert,
                                C_matrixes,
                                act,
                                list_meth[0],
                                indices_array_fix,
                                data_array_fix])
    
    # constant_inputs = ray.put([4,
    #                             4,
    #                             4,
    #                             4,
    #                             4,
    #                             4])
    
    print("Started parallel computations")
    results_ray = [compute_stochastic_lcas_1worker.remote(constant_inputs,array_for_dp,chunk_size) for array_for_dp,chunk_size in zip(list_arrays_for_datapackages,list_chunk_sizes)]
    results = ray.get(results_ray)
    end_time = time.time()
    
    ray.shutdown()




    # Combine results from all chunks
    combined_results = [np.vstack([result[i] for result in results]) for i in range(len(list_meth))]
    list_tables = [pd.DataFrame(combined_results[i], columns=[fu_[1] for fu_ in list_fu]) for i in range(len(list_meth))]

    # # The last row actually corresponds to the first one
    # for table_index in range(len(list_tables)):
    #     # Add the last row as first row
    #     list_tables[table_index] = pd.concat([pd.DataFrame(list_tables[table_index].iloc[-1]).T, list_tables[table_index]], ignore_index=True)
    #     # Delete the last row 
    #     list_tables[table_index] = list_tables[table_index].drop(list_tables[table_index].index[-1])

    print(f"Total computation time: {end_time - start_time} seconds")
    
    
    return list_tables   
    
    
    
    
    
    
    
def divide_dict(dict_total, x):
    
    ###
    """ Tahes the dictionnary containing the stochastic values for each parameter
    and divides into  a list of dictionnaries with chunks of the stochastic samples, to be processed in parallel """
    ###
    
    # Initialize the list of dictionaries to be returned
    dicts_list = [{} for _ in range(x)]

    # Iterate over each variable in the original dictionary
    for var, data in dict_total.items():
        values = data["values"]
        n = len(values)
        chunk_size = n // x
        
        
        remainder = n % x
        
        # Split values into x chunks
        chunks = [values[i*chunk_size + min(i, remainder):(i+1)*chunk_size + min(i+1, remainder)] for i in range(x)]
        
        # Assign chunks to the new dictionaries
        for i in range(x):
            dicts_list[i][var] = {"values": chunks[i]}
    
    
    list_chunk_sizes=[len(chunk) for chunk in chunks]

    
    return dicts_list,list_chunk_sizes















################################################################################
"""PRIM"""
###############################################################################








#OK
def prepare_outputs_all_main(list_tables,
                    sample_dataframe,
                    delta,
                    type_delta,
                    mode,
                    act1,
                    act2,
                    name_meltcolumn,
                    colnames_switch):
    
    """Calls the function which process the LCA inputs and outputs to return the necessary inputs for PRIM. A list with one element per impact category """
    
    
    list_processed_outputs = [prep_for_prim_1_categ_all_main(table,sample_dataframe,delta,mode,type_delta,act1,act2,name_meltcolumn,colnames_switch) for table in list_tables] 

    return list_processed_outputs







#ok
def prepare_outputs_all_categ_all_main(list_tables,
                    sample_dataframe,
                    delta,
                    type_delta,
                    mode,
                    act1,
                    act2,
                    list_meth,
                    index_meth_conditions,
                    name_meltcolumn,
                    colnames_switch):
    
    """Calls the function that Processes the LCA inputs and outputs to return the necessary inputs for PRIM with success for all categories together. """

    
    
    table_parameters,y_crop,y_elec, y_all_main_products, recombine_table = prep_for_prim_all_categ_all_main(list_tables,
                                sample_dataframe, 
                                delta, 
                                mode,
                                type_delta,
                                act1,
                                act2,
                                list_meth,
                                index_meth_conditions,
                                name_meltcolumn,
                                colnames_switch)  




    return table_parameters,y_crop,y_elec, y_all_main_products, recombine_table



# ok 
def get_box_data(box_):
    
    """Function which collects the data about a PRIM box"""    
    
    box_data = box_.inspect(style="data")
    box_data_series = box_data[0][0]
    dens = box_data_series["density"] 
    cover = box_data_series["coverage"] 
    mass = box_data_series["mass"]
    mean = box_data_series["mean"]
    
    boundaries_box = box_data[0][1]
    boundaries_box_df = boundaries_box.T.reset_index()
    
    
    return dens,cover,mass,mean,boundaries_box_df




    

# ok 
def prep_for_prim_all_categ_all_main(mc_results_all_categ,
                            sample_dataframe, 
                            delta, 
                            mode,
                            type_delta,
                            act1,
                            act2,
                            list_meth,
                            index_meth_conditions,
                            name_meltcolumn,
                            colnames_switch):  #[crop, elec]

    """ Processes the LCA inputs and outputs to return the inputs for PRIm (LCA input and yes/no success arrays. 
    Returns sucess arrays for crop as a product, elec as main, or both together """
    
    #wheat_int=corres_int_for_category(colnames_switch,"wheat_switch")
    wheat_int="wheat_switch" # Here we keep the whole name for the version with categories
    soy_int="soy_switch"
    alfalfa_int="alfalfa_switch"
    
    
    table_parameters = sample_dataframe.reset_index(drop="False")
    recombine_table = table_parameters.copy()
    
    columns_of_partial_success_crop=[]
    columns_of_partial_success_elec=[]

    columns_of_necessary_partial_success_crop = []
    columns_of_necessary_partial_success_elec = []
    
    
    
    for meth_index in range(len(mc_results_all_categ)):
        
        name_meth= list_meth[meth_index][2]  #+list_meth[meth_index][0][-8:]
        mc_results = mc_results_all_categ[meth_index]
        mc_results_clean = mc_results.copy()
        mc_results_clean["difference_elec_main"] = mc_results_clean[act1]-mc_results_clean[act2]
        mc_results_clean["percent_modif_elec_main"] = mc_results_clean["difference_elec_main"]/abs(mc_results_clean[act2])
        
        
        # When crop is the main product, we need to compare AVS with the corresponding tyope of crop
        dif_wheat = mc_results_clean[recombine_table[name_meltcolumn]==wheat_int]["AVS_crop_main"]-mc_results_clean[recombine_table[name_meltcolumn]==wheat_int]['wheat_fr_ref']
        wheat_imp = mc_results_clean[recombine_table[name_meltcolumn]==wheat_int]['wheat_fr_ref']
        dif_soy = mc_results_clean[recombine_table[name_meltcolumn]==soy_int]["AVS_crop_main"]-mc_results_clean[recombine_table[name_meltcolumn]==soy_int]['soy_ch_ref']
        soy_imp = mc_results_clean[recombine_table[name_meltcolumn]==soy_int]['soy_ch_ref']

        dif_alfalfa = mc_results_clean[recombine_table[name_meltcolumn]==alfalfa_int]["AVS_crop_main"]-mc_results_clean[recombine_table[name_meltcolumn]==alfalfa_int]['alfalfa_ch_ref']
        alfalfa_imp = mc_results_clean[recombine_table[name_meltcolumn]==alfalfa_int]['alfalfa_ch_ref']

        # Concatenate the series while aligning them based on indexes
        dif_all = pd.concat([dif_wheat, dif_soy, dif_alfalfa])
        corresponding_imp_all = pd.concat([wheat_imp, soy_imp, alfalfa_imp])
        

        # Sort the index to maintain the correct order
        dif_all.sort_index(inplace=True)
        corresponding_imp_all.sort_index(inplace=True)
        
        mc_results_clean["difference_crop_main"]=dif_all
        mc_results_clean["corresponding_impact_crop_main"]=corresponding_imp_all

        mc_results_clean["percent_modif_crop_main"] = mc_results_clean["difference_crop_main"]/abs(mc_results_clean["corresponding_impact_crop_main"])
        
        if type_delta=="percent_modif":
            print("percent_modif")
            if mode == "sup":
                
                #(AVS - PV) / PV           
                #y_elec_main = (mc_results_clean[act1]-mc_results_clean[act2])/abs(mc_results_clean[act2]) > delta  # the cases of interest
                
                y_elec_main = mc_results_clean["percent_modif_elec_main"] > delta  # the cases of interest
               
                y_crop_main = mc_results_clean["percent_modif_crop_main"] > delta  # the cases of interest
     
                mc_results_clean["Success_elec_main"] = y_elec_main
                mc_results_clean["Success_crop_main"] = y_crop_main
                
        
            elif mode == "inf": 
                #y_elec_main = (mc_results_clean[act1]-mc_results_clean[act2])/abs(mc_results_clean[act2]) < -delta  # the cases of interest
    
                y_elec_main  =mc_results_clean["percent_modif_elec_main"]  < -delta  # the cases of interest
                
                y_crop_main = mc_results_clean["percent_modif_crop_main"]  < -delta  # the cases of interest
                
                mc_results_clean["Success_elec_main"] = y_elec_main
                mc_results_clean["Success_crop_main"] = y_crop_main
    
            else:
                sys.exit("Invalid mode of success")        
            
        
        
        
        mc_results_clean.columns = [colname+name_meth for colname in mc_results_clean.columns]
        #print(mc_results_clean.columns)
        
        columns_of_partial_success_crop.append("Success_crop_main"+name_meth)
        columns_of_partial_success_elec.append("Success_elec_main"+name_meth)
        
        if meth_index in index_meth_conditions:
            
            columns_of_necessary_partial_success_crop.append("Success_crop_main"+name_meth)
            columns_of_necessary_partial_success_elec.append("Success_elec_main"+name_meth)
            
            
        recombine_table = pd.concat([recombine_table,mc_results_clean], axis=1)

        
    #print("rrr",recombine_table.columns)

    # Now find conditiosn of total success
    
    number_successes_crop = sum(recombine_table[partial_success] for partial_success in columns_of_partial_success_crop)
    number_successes_elec = sum(recombine_table[partial_success] for partial_success in columns_of_partial_success_elec)

    number_necessary_successes_crop = sum(recombine_table[partial_success] for partial_success in columns_of_necessary_partial_success_crop)
    number_necessary_successes_elec = sum(recombine_table[partial_success] for partial_success in columns_of_necessary_partial_success_elec)

    recombine_table["number_successes_elec"] = number_successes_elec
    recombine_table["number_successes_crop"] = number_successes_crop

    recombine_table["number_necessary_successes_elec"] = number_necessary_successes_elec
    recombine_table["number_necessary_successes_crop"] = number_necessary_successes_crop

    y_crop = recombine_table["number_necessary_successes_crop"]==len(index_meth_conditions) # Successes for all categories
    y_elec = recombine_table["number_necessary_successes_elec"]==len(index_meth_conditions) # Successes for all categories
    
    # True only if True for both perspectives
    y_all_main_products = y_crop * y_elec
    
    recombine_table["Total_success_crop"] = y_crop
    recombine_table["Total_success_elec"] = y_elec
            
    recombine_table["Total_success_all"] = y_all_main_products
            
    # For manual pairplot
    

    return table_parameters,y_crop,y_elec, y_all_main_products, recombine_table







def corres_int_for_category(colnames_switch,name_categ):
    
    """Converts category names into integers"""
    
    corres_cat_colnames_switch = [i for i in enumerate(colnames_switch)]
    corres_int = [a[0] for a in corres_cat_colnames_switch if a[1]==name_categ][0]

    return corres_int

#ok
def prep_for_prim_1_categ_all_main(mc_results,sample_dataframe, delta, mode,type_delta,act1,act2,name_meltcolumn,colnames_switch):  # mode = inf or sup to the probability threshold

    """  Prepares input for prim. Used for all categories together"""
    
    
    mc_results_clean = mc_results.copy()
    
        

    # Prepare Parameters table
    table_parameters = sample_dataframe.reset_index(drop="False")
    
    recombine_table = pd.concat([table_parameters,mc_results_clean], axis=1)

    recombine_table["difference_elec_main"] = recombine_table[act1]-recombine_table[act2]  # elec AVS - elec PV
    recombine_table["percent_modif_elec_main"] = recombine_table["difference_elec_main"]/abs(recombine_table[act2])
        
    #wheat_int=corres_int_for_category(colnames_switch,"wheat_switch") #
    wheat_int="wheat_switch" # we keep the names of the categories for this version
    soy_int="soy_switch"
    alfalfa_int="alfalfa_switch"
    
    # When crop is the main product, we need to compare AVS with the corresponding tyope of crop
    dif_wheat = recombine_table[recombine_table[name_meltcolumn]==wheat_int]["AVS_crop_main"]-recombine_table[recombine_table[name_meltcolumn]==wheat_int]['wheat_fr_ref']
    wheat_imp = recombine_table[recombine_table[name_meltcolumn]==wheat_int]['wheat_fr_ref']
    dif_soy = recombine_table[recombine_table[name_meltcolumn]==soy_int]["AVS_crop_main"]-recombine_table[recombine_table[name_meltcolumn]==soy_int]['soy_ch_ref']
    soy_imp = recombine_table[recombine_table[name_meltcolumn]==soy_int]['soy_ch_ref']

    dif_alfalfa = recombine_table[recombine_table[name_meltcolumn]==alfalfa_int]["AVS_crop_main"]-recombine_table[recombine_table[name_meltcolumn]==alfalfa_int]['alfalfa_ch_ref']
    alfalfa_imp = recombine_table[recombine_table[name_meltcolumn]==alfalfa_int]['alfalfa_ch_ref']

    # Concatenate the series while aligning them based on indexes
    dif_all = pd.concat([dif_wheat, dif_soy, dif_alfalfa])
    corresponding_imp_all = pd.concat([wheat_imp, soy_imp, alfalfa_imp])
    

    # Sort the index to maintain the correct order
    dif_all.sort_index(inplace=True)
    corresponding_imp_all.sort_index(inplace=True)
    
    recombine_table["difference_crop_main"]=dif_all
    recombine_table["corresponding_impact_crop_main"]=corresponding_imp_all

    recombine_table["percent_modif_crop_main"] = recombine_table["difference_crop_main"]/abs(recombine_table["corresponding_impact_crop_main"])

    
    if type_delta=="percent_modif":
        
        print("percent_modif")
        if mode == "sup":
            
            #(AVS - PV) / PV           
            #y_elec_main = (recombine_table[act1]-recombine_table[act2])/abs(recombine_table[act2]) > delta  # the cases of interest
            
            y_elec_main = recombine_table["percent_modif_elec_main"] > delta  # the cases of interest
           
            y_crop_main = recombine_table["percent_modif_crop_main"] > delta  # the cases of interest
 
            recombine_table["Success_elec_main"] = y_elec_main
            recombine_table["Success_crop_main"] = y_crop_main
            
    
        elif mode == "inf": 
            #y_elec_main = (recombine_table[act1]-recombine_table[act2])/abs(recombine_table[act2]) < -delta  # the cases of interest

            y_elec_main = recombine_table["percent_modif_elec_main"] < -delta  # the cases of interest
            
            y_crop_main = recombine_table["percent_modif_crop_main"]  < -delta  # the cases of interest

            recombine_table["Success_elec_main"] = y_elec_main

            recombine_table["Success_crop_main"] = y_crop_main

        else:
            sys.exit("Invalid mode of success")              
        
        
        

            
            
    # For manual pairplot
    

    return table_parameters,y_elec_main,y_crop_main


















# PRIM PARALLEL



#
def find_boxes_and_collect(table_parameters,
                           y_elec_main,
                           y_crop_main,
                           number_of_points,
                           meth_code,
                           main,
                           min_mass,
                           tresh_density,
                           peel_alpha,
                           mode_prim,
                           max_boxes,
                           sample_dataframe,
                           plot):
    
    """ Applies PRIM and returns boxes """
    
    
    
    

    info_meth ={}
    list_boxes_limits_df =[]
    scenario_data={}
    
    
    if main=="elec":
        y=y_elec_main
        
    elif main=="crop":
        y=y_crop_main
        
    elif main=="both":
        y=y_crop_main * y_elec_main    
        
    

    # Filter and clean spaces
 
    len_list_box = 0 #Initialize with True
            

    list_box = []

    print("pp")

    # Call PRIM
    if sum(y)!=0 and sum(y)!=len(y):
        
        successes=True
        info_meth["Successes"]="Yes"

        prim_alg = prim.Prim(table_parameters,
                             y,
                             threshold=tresh_density,
                             peel_alpha=peel_alpha,
                             mass_min = min_mass,
                             update_function=mode_prim) 
        print("There were some successes")

    elif sum(y)!=0 and sum(y)==len(y):
        
        info_meth["Successes"]="All successes"

        successes=False

        print("All successes")

    else:
        
        info_meth["Successes"]="No success"
        #write_json_file(folderpath4+"/"+"info_meth"+meth_code, info_meth)

        successes=False

        print("No successes")            
        


    # Collect the found boxes
    if successes:
        for i in range(max_boxes):
            
            print("tt",i)
            try:
                list_box.append(prim_alg.find_box())
            except:
                print("Error with qp")   # Possible error that must be ignored
        
        list_box = [i for i in list_box if i is not None]
        
        len_list_box = len(list_box)
        #print("yyyyy",len_list_box)
        
        no_box=len_list_box==1

        inital_density = list_box[0].inspect(0,style="data")[0][0]["density"]
        
        
        
        info_meth["Number boxes"]=len_list_box # 1 means that no box was found
        info_meth["initial_density"]=inital_density
        info_meth["Number of True points"] = initial_number_true = sum(y)



        
        cover_tot = 0 #Initialize the coverage for all boxes
    
        
        #write_json_file(folderpath4+"/"+"info_meth"+meth_code, info_meth)

        
        #scenario_data["info_meth"] = info_meth
            
        list_boxes_limits_df.append(info_meth)
        
        
        for index_box in range(len(list_box)):  # 
            
            box = list_box[index_box]
            
            print("index_box",index_box)
            
            [dens,cover,mass,mean,boundaries_box_df] = get_box_data(box)
            
            cover_tot +=cover
            
            
            
           

            number_of_points_in_box = mass * number_of_points
            
            number_of_true_points_in_box = dens * number_of_points_in_box

            remaining_true_points =  initial_number_true - (cover_tot * initial_number_true)
            
            scenario_data = {"box": index_box, 
                             "density": dens,
                             "cover": cover,
                             "cover_cumul":cover_tot,
                             "number_of_points_in_box":number_of_points_in_box,
                             "number_of_true_points_in_box":number_of_true_points_in_box,
                             "number_of_remaining_true_points":remaining_true_points}
            
            for param_name in list(sample_dataframe.columns):

                if param_name in list(boundaries_box_df.columns): # if not constrained
                    #print(param_name)
                    
                    if not isinstance(sample_dataframe[param_name][0],str): 
                    
                        

                    
                         print("rr ",param_name)

                         scenario_data[param_name] = [boundaries_box_df[param_name][0],
                                                      boundaries_box_df[param_name][1],  
                                                      1 if pd.isna(boundaries_box_df[param_name][2]) else boundaries_box_df[param_name][2],
                                                      1 if pd.isna(boundaries_box_df[param_name][3]) else boundaries_box_df[param_name][3]]
                    else:  # If categories

                         scenario_data[param_name] = [list(boundaries_box_df[param_name][0]), # when categorical variables, 0 and 1 are the same set of selected categories. We convert them to lists.
                                                      list(boundaries_box_df[param_name][1]),  
                                                      1 if pd.isna(boundaries_box_df[param_name][2]) else boundaries_box_df[param_name][2],
                                                      1 if pd.isna(boundaries_box_df[param_name][3]) else boundaries_box_df[param_name][3]]
                        
                        
                else:  # NOT CONSTRAINED
                    
                    if not isinstance(sample_dataframe[param_name][0],str):
                    
                        #print(sample_dataframe[param_name][0])
                        scenario_data[param_name] = [min(sample_dataframe[param_name]), max(sample_dataframe[param_name]), 0, 0]
                        
                    else:
                        
                    
                        scenario_data[param_name] = [np.unique(sample_dataframe[param_name]).tolist(), np.unique(sample_dataframe[param_name]).tolist(), 0, 0] 
                        
                        
                        
            list_boxes_limits_df.append(scenario_data)
            
            
            if plot:
                #Plot box 
                plt.figure()
                plot = box.inspect(style='graph')
                   
    
                dens_trunc = '%.3f'%(dens)    
                inital_density_trunc  = '%.3f'%(inital_density) 
                cover_trunc = '%.3f'%(cover) 
                mass_trunc = '%.3f'%(mass) 
               
                name_box = "box_"+str(index_box)+"_"+code_boxes
                
                
                plot[0].suptitle(str(meth_code)+" "+"init dens="+inital_density_trunc +"  box dens="+dens_trunc +"  mass="+mass_trunc  +"  cover="+cover_trunc)
                plot[0].tight_layout()
                
                plot[0].savefig(folderpath2+"/"+name_box+".png",dpi=500,bbox_inches="tight")
                
            
            
                # Show peeling trajectory
                
                
                plt.figure()
                plot_peeling = box.show_tradeoff()
                plot_peeling.suptitle(str(meth_code)+" "+"init dens="+inital_density_trunc +"  box dens="+dens_trunc +"  mass="+mass_trunc  +"  cover="+cover_trunc)
                plot_peeling.tight_layout()
                
                plot_peeling.savefig(folderpath3+"/"+name_box+".png",dpi=500,bbox_inches="tight")
                
    
            
            
            
    
    

        print("OKK")


    return list_boxes_limits_df






#ok
@ray.remote
def parallel_PRIM_meth(constant_inputs,
                       res_for_meth,
                       meth
                       ):
    
    """Function which calls PRIM for all impact categories on the regionalized outputs,
    saves and plot discovered boxes"""  
    
    # def write_json_file(file_path, data):
    #     """
    #     Write data to a JSON file.
        
    #     Args:
    #     - file_path: The path of the JSON file to write.
    #     - data: The data to write to the JSON file.
    #     """
    #     with open(file_path, 'w') as json_file:
    #         json.dump(data, json_file, indent=4)
    #     print(f"Data written to '{file_path}' successfully.")
    
    
    [sample_dataframe,
    delta,
    type_delta,
    mode_compa,
    act1,
    act2,
    min_mass,
    tresh_density,
    peel_alpha,
    mode_prim,
    plot,
    index_meth_conditions,
    main,
    max_boxes,
    number_of_points]=constant_inputs
    
        
    meth_code = meth[-1]   # Collect the method code
    
    table_parameters = res_for_meth[0]
    y_elec_main = res_for_meth[1]
    y_crop_main = res_for_meth[2]
    
        
    
    
    
    list_boxes_limits_df = find_boxes_and_collect(table_parameters,
                               y_elec_main,
                               y_crop_main,
                               number_of_points,
                               meth_code,
                               main,
                               min_mass,
                               tresh_density,
                               peel_alpha,
                               mode_prim,
                               max_boxes,
                               sample_dataframe,
                               plot
                               )
    


    return list_boxes_limits_df





    





#ok
def go_over_list_PRIM_while_total_success_all_main_parallel(list_tables,
                              sample_dataframe,
                              delta,
                              type_delta,
                              mode_compa,
                              act1,
                              act2,
                              list_meth, 
                              min_mass,
                              tresh_density,
                              peel_alpha,
                              mode_prim,
                              plot,
                              index_meth_conditions,
                              main,
                              max_boxes,
                              name_meltcolumn,
                              colnames_switch,
                              fixedornot):
    
    """Function which calls PRIM for all impact categories on the regionalized outputs,
    saves and plot discovered boxes"""  
    
    

    
    """"First going through all categories separately"""
    
    
    number_of_points = sample_dataframe.shape[0]
    
    tables_for_PRIM= prepare_outputs_all_main(list_tables,
                        sample_dataframe,
                        delta,
                        type_delta,
                        mode_compa,
                        act1,
                        act2,
                        name_meltcolumn,
                        colnames_switch) 

    info_algo = {
        "min_mass": min_mass,
        "tresh_density": tresh_density,
        "peel_alpha": peel_alpha,
        "mode_prim": mode_prim,
        "shape sample":sample_dataframe.shape,
        "mode_compa": mode_compa,
        "mode_prim": mode_prim,
        "type_delta":type_delta,
        "Success Impact Categories":[list_meth[index] for index in index_meth_conditions],
        "main":main,
        "max_boxes":max_boxes
    }

    
    x = datetime.datetime.now()
    
    month=str(x.month)
    day=str(x.day)
    microsec=str(x.strftime("%f"))
    code_output_folder="ms"+str(min_mass)+"_ds"+str(tresh_density)+"_pm"+str(mode_prim)+"_pl"+str(peel_alpha)+"_"+month+"_"+day+"_"+microsec+"dta_"+str(delta)+"compa"+str(mode_compa)+str(main)+str(fixedornot)

    folderpath1="../PRIM_process/Boxes/"+code_output_folder
    folderpath2="../PRIM_process/Box pairplots/"+code_output_folder
    folderpath3="../PRIM_process/Peeling/"+code_output_folder
    folderpath4= "../PRIM_process/info/"+code_output_folder

    folderpaths = [folderpath1,
      folderpath2,
      folderpath3,
      folderpath4]           

    [create_folder_if_not_exists(folder_path) for folder_path in folderpaths]


    ray.shutdown()
    
    #(runtime_env={"working_dir": "C:\local\pjouannais\Code et autres\BW 2.5\Scripts model\Scripts soft links"}
    
    # Get the absolute path of the current working directory
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)


    ray.init(runtime_env={"working_dir": current_directory},configure_logging=False, log_to_driver=False)
    
    #The inputs that are constant to all tasks/workers
    constant_inputs = ray.put([sample_dataframe,
                                delta,
                                type_delta,
                                mode_compa,
                                act1,
                                act2,
                                min_mass,
                                tresh_density,
                                peel_alpha,
                                mode_prim,
                                plot,
                                index_meth_conditions,
                                main,
                                max_boxes,
                                number_of_points])
    
    
    # Simulate
    list_boxes_limits_df_all_meth = ray.get([parallel_PRIM_meth.remote(constant_inputs,res_for_meth,meth) for res_for_meth,meth in zip(tables_for_PRIM,list_meth)])
    

    
    
    #list_boxes_limits_df_all_meth = [sub[0] for sub in ray_results ] 


        
    # Second, Study complete success
    #table_parameters_all,y_all, recombine_table_all
    print("Now working on the complete success")
    
    table_parameters_all,y_crop_all,y_elec_all, y_all_main_products, recombine_table_all = prepare_outputs_all_categ_all_main(list_tables,
                        sample_dataframe,
                        delta,
                        type_delta,
                        mode_compa,
                        act1,
                        act2,
                        list_meth,
                        index_meth_conditions,
                        name_meltcolumn,
                        colnames_switch)
    


    list_boxes_limits_df_total =  find_boxes_and_collect(table_parameters_all,
                               y_elec_all,
                               y_crop_all,
                               number_of_points,
                               "TOTAL",
                               main,
                               min_mass,
                               tresh_density,
                               peel_alpha,
                               mode_prim,
                               max_boxes,
                               sample_dataframe,
                               plot)



    return list_boxes_limits_df_all_meth,list_boxes_limits_df_total,recombine_table_all











#ok
def apply_prim_different_settings_all_main_parallel(list_alpha,
                                  list_delta,
                                  list_mode_compa,
                                  list_densities,
                                  list_mode_prim,
                                  list_tables,
                                  sample_dataframe,
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
                                  fixedornot):
    
    """Function to call which runs the prim algorithm with different setings"""
    dict_results = {}
    
    for peel_alpha in list_alpha:
        for index in range(len(list_delta)):
            
            delta=list_delta[index]
            mode_compa=list_mode_compa[index]
    
            print("delta ===",delta)
        
            for tresh_density in list_densities:
                print("tresh density ===",tresh_density)
               
                for mode_prim in list_mode_prim:
                    [
                     list_boxes_limits_df_all_meth,
                     list_boxes_limits_df_total,
                     recombine_table_all]=go_over_list_PRIM_while_total_success_all_main_parallel(list_tables,
                                                  sample_dataframe,
                                                  delta,
                                                  type_delta,
                                                  mode_compa,
                                                  act1,
                                                  act2,
                                                  list_meth, 
                                                  min_mass,
                                                  tresh_density,
                                                  peel_alpha,
                                                  mode_prim,
                                                  plot,
                                                  index_meth_conditions,
                                                  main,
                                                  max_boxes,
                                                  name_meltcolumn,
                                                  colnames_switch,
                                                  fixedornot)
                    
                    dict_results["delta_"+str(delta)
                                 +"mode_compa_"+str(mode_compa)
                                 +"tresh_density_"+str(tresh_density)
                                 +"peel_alpha_"+str(peel_alpha)
                                 +"mode_prim_"+str(mode_prim)]  =[
                                  list_boxes_limits_df_all_meth,
                            
                                  list_boxes_limits_df_total,
                                  recombine_table_all]
                    
                    
                    x = datetime.datetime.now()
                    
                    month=str(x.month)
                    day=str(x.day)
                    microsec=str(x.strftime("%f"))
                                 
            
            
            
                    name_file_list_boxes_limits_df_all_meth ='list_boxes_limits_df_all_meth'+"_"+month+"_"+day+"_"+microsec+"delta="+str(delta)+"mode_compa="+str(mode_compa)+"mode_prim="+str(mode_prim)+"main="+str(main)
                    name_file_list_boxes_limits_df_total ='list_boxes_limits_df_total'+"_"+month+"_"+day+"_"+microsec+"delta="+str(delta)+"mode_compa="+str(mode_compa)+"mode_prim="+str(mode_prim)+"main="+str(main)
                    name_file_recombine_table_all ='recombine_table_all'+"_"+month+"_"+day+"_"+microsec+"delta="+str(delta)+"mode_compa="+str(mode_compa)+"mode_prim="+str(mode_prim)+"main="+str(main)
            
                   
                    
                    export_pickle_2(list_boxes_limits_df_all_meth, name_file_list_boxes_limits_df_all_meth, "resultsintermediate")
                    export_pickle_2(list_boxes_limits_df_total, name_file_list_boxes_limits_df_total, "resultsintermediate")
                    export_pickle_2(recombine_table_all, name_file_recombine_table_all, "resultsintermediate")
                    
                    recombine_table_all.to_csv("../resultsintermediate/"+name_file_recombine_table_all+'.csv', sep=';', encoding='utf-8')
        
    return dict_results









