# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:56:31 2024

@author: pjouannais

Scrit to export boxes into an excel file.
Update import paths l63
"""



from Main_functionsimport import *
import Parameters_and_functions
from Parameters_and_functions  import *
import matrix_utils as mu
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

import pandas as pd



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


import datetime
import os

import json

import ray

import re

import copy


###
""" Import"""
###

#ema_logging.log_to_stderr(ema_logging.INFO)


listmeth= importpickle("../resultsintermediate/server/listmethpara400k_7_23_848065size=400000.pkl")

   
# Higher samplinjg

dict_res_checkboxes_both400Khs = importpickle("../PRIM_process/Boxes/server/400klhs/hs/checkboxes_both8_7_497758size=2000.pkl")
dict_res_checkboxes_elec400Khs = importpickle("../PRIM_process/Boxes/server/400klhs/hs/checkboxes_elec8_7_219458size=2000.pkl")
dict_res_checkboxes_crop400Khs = importpickle("../PRIM_process/Boxes/server/400klhs/hs/checkboxes_crop8_9_988126size=2000.pkl")


      
    

# Import dict res

dictres_import_both_varpv_validboxes= importpickle("../PRIM_process/dict_results/server/400K/dict_results_all_main_both_400k_lhsvalid.pkl")
dictres_import_crop_varpv_validboxes= importpickle("../PRIM_process/dict_results/server/400K/dict_results_all_main_crop_400k_lhsvalid.pkl")
dictres_import_elec_varpv_validboxes= importpickle("../PRIM_process/dict_results/server/400K/dict_results_all_main_elec_400k_lhsvalid.pkl")





#####
""" Functions"""


def update_density(dict_checked_boxes_ori,
                   list_actual_boxes_ori,):
    
    list_actual_boxes = copy.deepcopy(list_actual_boxes_ori)
    dict_checked_boxes = copy.deepcopy(dict_checked_boxes_ori)
    
    dict_checked_boxes.pop("info_init")
    
    # initialize new key in the boxes dict
    for box in list_actual_boxes:
        
        box["double_checked_density"]="Not tested"
    
    for key in dict_checked_boxes:
        
        double_checked_dens = dict_checked_boxes[key]
        
        # Collect the corresponding box
        
        for box in list_actual_boxes:
            
            if box["box"]== key:
                box["double_checked_density"]=double_checked_dens
    return list_actual_boxes    


def update_actual_densities(dict_checkboxes_ori,
                     dict_boxes_ori):
    
    dict_checkboxes = copy.deepcopy(dict_checkboxes_ori)
    dict_boxes = copy.deepcopy(dict_boxes_ori)
    
    
    for key in dict_checkboxes.keys():
        
        
        dict_res_checkboxes =  dict_checkboxes[key]        
        dict_res_boxes =  dict_boxes[key]
        
        check_boxes_res_total = dict_res_checkboxes["Total"]
        list_check_boxes_res_meth = dict_res_checkboxes["Each impact category"]
        
        
        boxes_total = dict_res_boxes[1]
        list_boxes_meth = dict_res_boxes[0]
        
        print("len boxes",len(boxes_total))
        
        info= boxes_total.pop(0)


        updated_boxes_total = update_density(check_boxes_res_total,boxes_total)
        
        # here add actual densitites
        
        
        
        updated_boxes_total = [info] + updated_boxes_total
        
        list_updated_boxes_meth = []
        for meth_index in range(len(list_boxes_meth)):
            
            boxes_meth = list_boxes_meth[meth_index]
            check_boxes_meth = list_check_boxes_res_meth[meth_index]
 
            info = boxes_meth.pop(0)
            
            
            updated_boxes_meth = update_density(check_boxes_meth,boxes_meth)

            
            list_updated_boxes_meth.append([info]+updated_boxes_meth)
        
        
        dict_boxes[key][1] = updated_boxes_total
        dict_boxes[key][0] = list_updated_boxes_meth

    return dict_boxes




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
 

def export_combined_boxes_to_excel(dict_res_elec, dict_res_crop, dict_res_both, listmeth, filename):
    """
    Export data from three different density dictionaries into a single Excel file,
    where each sheet contains data from all three dictionaries, with each data section
    separated by 3 empty lines.
    
    Args:
    - dict_res_elec: Dictionary containing results for electricity.
    - dict_res_crop: Dictionary containing results for crop.
    - dict_res_both: Dictionary containing results for both.
    - listmeth: List of method names used for categorizing data.
    - filename: Name of the output Excel file.
    """
    # We only pop info in elec so that it's never selected in the other dicts
    dict_res_elec.pop("info")

    def get_combined_dict():
        """Utility function to create a combined dictionary of the three datasets."""
        return {'Electricity': dict_res_elec, 'Crop': dict_res_crop, 'Both': dict_res_both}

    combined_dict = get_combined_dict()

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for key in dict_res_elec.keys():
            # Initialize starting row for each sheet
            startrowtotal = 0
            list_startrow_meth = [0 for _ in range(len(listmeth))]
            for dict_name, dict_res in combined_dict.items():
                list_list_boxes_categories = dict_res[key][0]
                list_boxes_total = dict_res[key][1]

                # Extract comparison type and delta value for labeling
                compa, delta_value = extract_compa_delta_main(key)
                delta_value = round(delta_value, 2)

                if delta_value == 0 and compa == "sup":
                    write_delta_compa = "Impact(AVS) sup to Impact(Conv)"
                    short_write = "sup to Conv"
                elif delta_value == 0 and compa == "inf":
                    write_delta_compa = "Impact(AVS) inf to Impact(Conv)"
                    short_write = "inf to Conv"
                elif delta_value == 0.33 and compa == "inf":
                    write_delta_compa = "Impact(AVS) inf to 0.67 Impact(Conv)"
                    short_write = "inf to 0.67 Conv"
                elif delta_value == 0.33 and compa == "sup":
                    write_delta_compa = "Impact(AVS) sup to 1.33 Impact(Conv)"
                    short_write = "sup to 1.33 Conv"
                else:
                    print("Unrecognized delta value and comparison type:", delta_value, compa)

                # Prepare the sheet name and description for the "Total Success" category
                category = "Totalsuccess"
                description_total_boxes = category + "\n" + write_delta_compa + "\n Sampling result, prim boxes limits for AVS, together with qp values.\n For a numerical parameter: [lower limit, upper limit, qp value lower, qp value higher] \n For a categorical parameter: [Values in the box, qp value]"
                sheet_name_total = category + " " + short_write

                # Write data for "Total Success" category
                startrowtotal = export_list_boxes_to_excel(list_boxes_total, f"{dict_name} - {description_total_boxes}", writer, sheet_name_total, startrowtotal)
                startrowtotal += 3  # Adding 3 empty lines after each section

                # Now handle individual methods
                for meth_index in range(len(listmeth)):
                    list_boxes_meth = list_list_boxes_categories[meth_index]
                    category = listmeth[meth_index][-1]
                    category_accronym = extract_meth_accronym(category)
                    description = category_accronym + "\n" + write_delta_compa + "\n Sampling result, prim boxes limits for AVS, together with qp values.\n For a numerical parameter: [lower limit, upper limit, qp value lower, qp value higher] \n For a categorical parameter: [Values in the box, qp value]"
                    sheet_name = category_accronym + " " + short_write
                    
                    # Write data for individual methods
                    list_startrow_meth[meth_index] = export_list_boxes_to_excel(list_boxes_meth, f"{dict_name} - {description}", writer, sheet_name, list_startrow_meth[meth_index])
                    list_startrow_meth[meth_index] += 3  # Adding 3 empty lines after each section

def export_list_boxes_to_excel(list_boxes, description, writer, sheet_name, startrow):
    """
    Exports a list of boxes to an Excel sheet starting from a specified row.
    Returns the next starting row after writing.
    """
    info_total = list_boxes.pop(0)  # Get the 'info' row
    info_df = pd.DataFrame.from_dict(info_total, orient='index').T

    # Write the description first
    worksheet = writer.sheets[sheet_name] if sheet_name in writer.sheets else writer.book.add_worksheet(sheet_name)
    worksheet.write(startrow, 0, description)
    startrow += 3  # Adding space after description
    
    # Write small DataFrame (info) at the specified row
    info_df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
    startrow += len(info_df) + 2  # Adding space after the small DataFrame

    # Write main DataFrame with the boxes
    export_boxes_to_excel(list_boxes, writer, sheet_name, startrow)

    # Calculate the new start row after writing the main DataFrame
    new_startrow = startrow + len(list_boxes) + 2
    return new_startrow

def export_boxes_to_excel(boxlimits_list, writer, sheet_name, startrow):
    """
    Exports box limits to an Excel sheet starting from a specified row.
    """
    # Initialize a dictionary to store the rows of data
    data = {}

    for i, boxlimits in enumerate(boxlimits_list):
        cover = boxlimits.pop("cover")
        cover_cumul = boxlimits.pop("cover_cumul")
        density = boxlimits.pop("density")
        number_of_points_in_box = boxlimits.pop("number_of_points_in_box")
        number_of_remaining_true_points = boxlimits.pop("number_of_remaining_true_points")
        number_of_true_points_in_box = boxlimits.pop("number_of_true_points_in_box")
        double_checked_dens = boxlimits.pop("double_checked_density")
        box = boxlimits.pop("box")

        row = {}
        for param, limits in boxlimits.items():
            if isinstance(limits[0], list):  # Check if the parameter is categorical
                allowed_values = "; ".join(limits[0])  # Using semicolon to avoid issues with commas in CSV
                qp_value = limits[2]
                row[f"{param} Allowed Values"] = allowed_values
                row[f"{param} qp"] = qp_value
            else:  # Numerical limits
                lower_limit, upper_limit, qp_low, qp_high = limits[0], limits[1], limits[2], limits[3]
                row[f"{param} low"] = lower_limit
                row[f"{param} up"] = upper_limit
                row[f"{param} qp low"] = qp_low
                row[f"{param} qp up"] = qp_high

        row["cover"] = round(cover, 5)
        row["cover_cumul"] = round(cover_cumul, 5)
        row["Points in box"] = number_of_points_in_box
        row["density"] = density
        row["Double-checked density"] = double_checked_dens
        
        data[f"Box {i+1}"] = row
    
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.rename(columns={'index': 'Box'}, inplace=True)
    
    # Write main DataFrame
    df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)




###
"""Run and export"""
###


updated_density_dict_elec = update_actual_densities(dict_res_checkboxes_elec400Khs,
                     dictres_import_elec_varpv_validboxes)        

updated_density_dict_crop = update_actual_densities(dict_res_checkboxes_crop400Khs,
                     dictres_import_crop_varpv_validboxes)  

updated_density_dict_both = update_actual_densities(dict_res_checkboxes_both400Khs,
                    dictres_import_both_varpv_validboxes) 



export_combined_boxes_to_excel(updated_density_dict_elec, updated_density_dict_crop, updated_density_dict_both, listmeth, "PRIM_process/Numerical_Catalog_boxes.xlsx")
