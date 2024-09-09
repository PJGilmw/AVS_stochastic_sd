# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:39:47 2024

@author: pjouannais

Script which double-checks the boxes found by PRIM and returns and exports updated boxes dictionnaries keeping only the ones that were validated above a certain density treshold.
The whole script must be executed after updating the import and export paths if necessary (l75 and l255).

threshold
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



### 
""" Choose actual density treshold"""
###
threshold =0.8
   
####
""" Import the dictionnaires """
###

### Dictionnaries of actual double-checked densitites for the boxes

dict_res_checkboxes_both400Khs = importpickle("../PRIM_process/Boxes/server/400klhs/hs/checkboxes_both8_7_497758size=2000.pkl")
dict_res_checkboxes_elec400Khs = importpickle("../PRIM_process/Boxes/server/400klhs/hs/checkboxes_elec8_7_219458size=2000.pkl")
dict_res_checkboxes_crop400Khs = importpickle("../PRIM_process/Boxes/server/400klhs/hs/checkboxes_crop8_9_988126size=2000.pkl")


### Dictionnaries of actual double-checked densitites for the boxes

dictres_import_both_varpv= importpickle("../PRIM_process/dict_results/server/400K/dict_results_all_main_both_400k82077674mainboth.pkl")
dictres_import_crop_varpv= importpickle("../PRIM_process/dict_results/server/400K/dict_results_all_main_crop_400k726233597maincrop.pkl")
dictres_import_elec_varpv= importpickle("../PRIM_process/dict_results/server/400K/dict_results_all_main_elec_400k728509021mainelec.pkl")




###
"""Functions """

def boxes_to_keep(dict_boxes_check_ori,threshold):
    
    """ Returns the indexes of the boxes that meth the density treshold condition"""
    
    dict_boxes_check = copy.deepcopy(dict_boxes_check_ori)

    #print(dict_boxes_check)
    info_init= dict_boxes_check.pop("info_init")

    tested_boxes = [key for key in dict_boxes_check.keys() ]
    
    print("rrrrrrr",tested_boxes)
    
    if len(tested_boxes)==1:
        
        step=1
    else:
        step = tested_boxes[1] - tested_boxes[0]
    
    if step == 1: # Then all found boxes have been assessed: less boxes than the sampling rate
       
        boxes_to_keep = [key for key in dict_boxes_check.keys() if dict_boxes_check[key] >=threshold]
        
        return boxes_to_keep
        
    else: # There were enough boxes so the samplign rate has been used
        
        last_four =  tested_boxes[-4:]
        
        boxes_to_keep_last_four = [key for key in last_four if dict_boxes_check[key] >=threshold]
        
        #other_boxes = [dict_boxes_check[key] for key in tested_boxes[-4:]] tested_boxes[:-4]    
        other_boxes = tested_boxes[:-4]
        
        boxes_to_keep_other_boxes = [key for key in other_boxes if dict_boxes_check[key] >=threshold]
        
        print("boxes_to_keep_other_boxes",boxes_to_keep_other_boxes)
        
        if len(boxes_to_keep_other_boxes)>0:
            
            threshold_index = boxes_to_keep_other_boxes[-1] #We initialize with the Last one of the other boxes
         
        else: # if none of the tested boxes with the sampling rate were above the threshold
            return boxes_to_keep_last_four
        
        
        for index in range(len(boxes_to_keep_other_boxes)-2):
            
            if boxes_to_keep_other_boxes[index+1]-boxes_to_keep_other_boxes[index] !=step: #there has been a gap
                
                print("kkk",index,boxes_to_keep_other_boxes[index]+1)
                
                
                
                return list(range(0,boxes_to_keep_other_boxes[index]+1)) + boxes_to_keep_last_four
        
        # Otherwise return all the boxes until the last one of the sampled ones, and the valid ones among the last four
        
        print("list(range(0,threshold_index)) + boxes_to_keep_last_four",list(range(0,threshold_index)) + boxes_to_keep_last_four)
        return list(range(0,threshold_index+1)) + boxes_to_keep_last_four
     


def update_cumul(list_actual_boxes):
    """ Updates the cumulated cover of double-checked kept boxes"""
    
    cumulcover=0
    
    for box in list_actual_boxes:
        
        cumulcover+=box["cover"]

        box["cover_cumul"]=cumulcover
        
    return list_actual_boxes


def collectboxnumber(dict_checkboxes_ori,
                     dict_boxes_ori,
                     tresh_actual):
    """ Returns an updated dictionnary of results with only the boxes that were validated after double-checking"""

    
    dict_checkboxes = copy.deepcopy(dict_checkboxes_ori)
    dict_boxes = copy.deepcopy(dict_boxes_ori)
    
    for key in dict_checkboxes.keys():
        
        dict_res_checkboxes =  dict_checkboxes[key]        
        dict_res_boxes =  dict_boxes[key]
        
        
        boxes_total = dict_res_boxes[1]
        list_boxes_meth = dict_res_boxes[0]
        
        print("len boxes",len(boxes_total))
        
        info= boxes_total.pop(0)

        check_boxes_res_total = dict_res_checkboxes["Total"]
        list_check_boxes_res_meth = dict_res_checkboxes["Each impact category"]
        
        print("TOTAL")
        
        boxes_to_keep_total = boxes_to_keep(check_boxes_res_total,tresh_actual)
        
        #  Total
        
        #print(boxes_total[1])
        print("OOOOO")
        print((len(boxes_total)))
              
        kept_boxes_total = [boxes_total[index] for index in range(len(boxes_total)) if boxes_total[index]["box"] in boxes_to_keep_total]
        
        kept_boxes_total = update_cumul(kept_boxes_total)
        
        # here add actual densitites
        
        
        
        kept_boxes_total = [info] + kept_boxes_total
        
        list_kept_boxes_meth = []
        for meth_index in range(len(list_boxes_meth)):
            
            boxes_meth = list_boxes_meth[meth_index]
            check_boxes_meth = list_check_boxes_res_meth[meth_index]
 
            boxes_to_keep_meth = boxes_to_keep(check_boxes_meth,tresh_actual)

            info = boxes_meth.pop(0)
            
            kept_boxes_meth = [boxes_meth[index] for index in range(len(boxes_meth)) if boxes_meth[index]["box"] in boxes_to_keep_meth]
            
            kept_boxes_meth = update_cumul(kept_boxes_meth)

            
            list_kept_boxes_meth.append([info]+kept_boxes_meth)
        
        
        dict_boxes[key][1] = kept_boxes_total
        dict_boxes[key][0] = list_kept_boxes_meth

    return dict_boxes



###
""" Run functions and exports results"""
###

correct_elec_boxes = collectboxnumber(dict_res_checkboxes_elec400Khs,
                     dictres_import_elec_varpv,
                     threshold)        

correct_crop_boxes = collectboxnumber(dict_res_checkboxes_crop400Khs,
                     dictres_import_crop_varpv,
                     threshold)  

correct_both_boxes = collectboxnumber(dict_res_checkboxes_both400Khs,
                     dictres_import_both_varpv,
                     threshold) 

export_pickle_2(correct_elec_boxes, "dict_results_all_main_elec_400k_lhsvalid2", "PRIM_process/dict_results/server/400K")

export_pickle_2(correct_crop_boxes, "dict_results_all_main_crop_400k_lhsvalid2", "PRIM_process/dict_results/server/400K")

export_pickle_2(correct_both_boxes, "dict_results_all_main_both_400k_lhsvalid2", "PRIM_process/dict_results/server/400K")