# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:50:12 2024

@author: pierre.jouannais

Script that starts setting up the foreground database.
To be executed as a whole.

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
import sys



bd.projects.set_current('AVS')


Ecoinvent = bd.Database('ecoinvent-3.10-consequential')

Biosphere = bd.Database('ecoinvent-3.10-biosphere')



          
# Delete databases before building them again 
if "additional_biosphere_multi_categories" in list(bd.databases):
    del bd.databases["additional_biosphere_multi_categories"]

if "foregroundAVS" in list(bd.databases):
    del bd.databases["foregroundAVS"]

  
if "additional_biosphere" in list(bd.databases):
    del bd.databases["additional_biosphere"]
      




# Creates the foreground database

foregroundAVS = bd.Database('foregroundAVS')


foregroundAVS.write({
    ("foregroundAVS", "AVS_crop_main"): {  
        'name': 'AVS_crop_main',
        'unit': 'kg',
        'exchanges': [{
                'input': ('foregroundAVS', 'AVS_crop_main'),
                'amount': 1,
                'unit': 'kg',
                'type': 'production'}]},
    
    ("foregroundAVS", "AVS_elec_main"): {  
        'name': 'AVS_elec_main',
        'unit': 'kWh',
        'exchanges': [{
                'input': ('foregroundAVS', 'AVS_elec_main'),
                'amount': 1,
                'unit': 'kWh',
                'type': 'production'}]},
    
    ("foregroundAVS", "PV_ref"): {  
        'name': 'PV_ref',
        'unit': 'kWh',
        'exchanges': [{
                'input': ('foregroundAVS', 'PV_ref'),
                'amount': 1,
                'unit': 'kWh',
                'type': 'production'}]
        },   
    
    
    ("foregroundAVS", "LUCmarket_AVS"): {  
        'name': 'LUCmarket_AVS',
        'unit': 'ha.y',
        'exchanges': [{
                'input': ('foregroundAVS', 'LUCmarket_AVS'),
                'amount': 1,
                'unit': 'ha.y',
                'type': 'production'}]
        },
    
    ("foregroundAVS", "LUCmarket_PVref"): {  
        'name': 'LUCmarket_PVref',
        'unit': 'ha.y',
        'exchanges': [{
                'input': ('foregroundAVS', 'LUCmarket_PVref'),
                'amount': 1,
                'unit': 'ha.y',
                'type': 'production'}]
        },
    
    ("foregroundAVS", "LUCmarket_cropref"): {  
        'name': 'LUCmarket_cropref',
        'unit': 'ha.y',
        'exchanges': [{
                'input': ('foregroundAVS', 'LUCmarket_cropref'),
                'amount': 1,
                'unit': 'ha.y',
                'type': 'production'}]
        },
    
    ("foregroundAVS", "iluc"): {  
        'name': 'iluc',
        'unit': 'ha.y',
        'exchanges': [{
                'input': ('foregroundAVS', 'iluc'),
                'amount': 1,
                'unit': 'ha.y',
                'type': 'production'}]
        },
    
    ("foregroundAVS", "c_soil_accu"): { 
        'name': 'c_soil_accu',
        'unit': 'kg C',
        'exchanges': [{
                'input': ('foregroundAVS', 'c_soil_accu'),
                'amount': 1,
                'unit': 'ha.y',
                'type': 'production'}]
        }})

    
    

# Collects previously creates activities

AVS_elec_main = foregroundAVS.get("AVS_elec_main")
AVS_crop_main = foregroundAVS.get("AVS_crop_main")

PV_ref =foregroundAVS.get("PV_ref")

LUCmarket_PVref = foregroundAVS.get("LUCmarket_PVref")
LUCmarket_cropref = foregroundAVS.get("LUCmarket_cropref")
LUCmarket_AVS = foregroundAVS.get("LUCmarket_AVS")

iluc = foregroundAVS.get("iluc")

c_soil_accu = foregroundAVS.get("c_soil_accu")










# Collects the ecoinvent crop activities



wheat_fr = Ecoinvent.get(
    "98ad08a169bac30f68833a6261d73e73")     # Wheat grain fr



    
soy_ch = Ecoinvent.get(
    "c97395c63e38cf03e87f0e3763b1a9b7")     # soy  ch




alfalfa_ch = Ecoinvent.get(
    "971ad0c601b5bb02b7a16450c48b28d5")     # alfalfa grain ch






def delete_transfo_occu(act):
    
    """ Deletes the transformation and the occupation exhhanges of an activity, and returns the original transformation flow """
    
    for exc in list(act.exchanges()):
        if exc["type"]=="biosphere":
            
            actinput=Biosphere.get(exc["input"][1])

            if "Transformation, to" in actinput["name"]:
                
                transfo_amount = exc["amount"]
                
                exc.delete()
                
            elif "Transformation, from" in actinput["name"]:
                            
                exc.delete()   
                
            elif "Occupation" in actinput["name"]:
                            
                exc.delete() 
    act.save()     
       
    return act,transfo_amount



def collect_transfo(act):
    
    """  Collects the transformation amount of an act """
    
    for exc in list(act.exchanges()):
        if exc["type"]=="biosphere":
            
            actinput=Biosphere.get(exc["input"][1])

            if "Transformation, to" in actinput["name"]:
                
                transfo_amount = exc["amount"]
                
            
    return transfo_amount


def rescale_to_ha(act,prodrescale,transfo_amount):
    
    """ Rescales a farming activity to 1 ha. """
    
    for exc in list(act.exchanges()):
        if exc["type"]!="production" or prodrescale:
            exc["amount"] = exc["amount"]*10000/transfo_amount          
            exc.save()
            
    act.save()  
    
    return act


def prepare_act_agri(original_act):
    
    """ Creates the crop activitites ready to be used in the AVS model"""
    
    act_ref = original_act.copy() # The conv zctivity
    
    act_AVS_elec_main = original_act.copy() # the crop virtual activity for when elec is the main product of AVS
    act_AVS_crop_main = original_act.copy() # the crop virtual activity for when crop is the main product of AVS
    

    act_ref_ori_code = act_ref["code"]

    # Collect yield (via transfo)
    
    transfo_amount = collect_transfo(act_AVS_crop_main)
    
    # The AVS agri act produces 1 Unit of virtural hectare.
    # the crop production becomes an output (substitution)
    
    for exc in list(act_AVS_elec_main.exchanges()):
        if exc["type"]=="production":
            prod_amount=exc["amount"]
            exc["amount"]=1 
            exc["unit"] = "virtual hectare unit"
            exc.save()
            act_AVS_elec_main.new_exchange(amount=-prod_amount, input=act_ref,type="technosphere").save()
    

    # The AVS crop_main act produces 1 Unit of virtural hectare.
    # No substitution
    for exc in list(act_AVS_crop_main.exchanges()):
        if exc["type"]=="production":
            prod_amount=exc["amount"]
            exc["amount"]=1 
            exc["unit"] = "virtual hectare unit"
            exc.save()    
    
    
    act_AVS_elec_main["unit"] = "virtual hectare unit"
    act_AVS_elec_main.save()
    
    act_AVS_crop_main["unit"] = "virtual hectare unit"
    act_AVS_crop_main.save()
    
    
    
    
    # Rescale to 1 ha
    
    act_ref = rescale_to_ha(act_ref,True,transfo_amount)
    
    act_AVS_crop_main = rescale_to_ha(act_AVS_crop_main,False,transfo_amount)
    

  
    # This also rescales  act_AVS_elec_main to ha
    for exc in list(act_AVS_elec_main.exchanges()):
        if exc["type"]!="production":
            for exc2 in list(act_ref.exchanges()):
                if exc["input"]==exc2["input"]:
                    if exc2["input"][1]==act_ref_ori_code:
                        print(exc2["input"])
                        exc["amount"] = - exc2["amount"]         
                        exc.save()
                        
                    else:
                        print(exc2["input"])
                        exc["amount"] = exc2["amount"]         
                        exc.save()    
    act_AVS_elec_main.save()  
    
    
    
    return act_AVS_elec_main,act_ref,act_AVS_crop_main
    
# Creates the crop activities with the function

# Wheat

# The "virtual" act for AVS with elec main, conv crop act, "virtual" for act for AVS with elec main
wheat_fr_AVS_elec_main,wheat_fr_ref,wheat_fr_AVS_crop_main = prepare_act_agri(wheat_fr)  

wheat_fr_AVS_elec_main["database"] = 'foregroundAVS'
wheat_fr_AVS_elec_main["name"] = 'wheat_fr_AVS_elec_main'
wheat_fr_AVS_elec_main["code"] = 'wheat_fr_AVS_elec_main'
wheat_fr_AVS_elec_main.save()

wheat_fr_ref["database"] = 'foregroundAVS'
wheat_fr_ref["name"] = 'wheat_fr_ref'
wheat_fr_ref["code"] = 'wheat_fr_ref'
wheat_fr_ref.save()

wheat_fr_AVS_crop_main["database"] = 'foregroundAVS'
wheat_fr_AVS_crop_main["name"] = 'wheat_fr_AVS_crop_main'
wheat_fr_AVS_crop_main["code"] = 'wheat_fr_AVS_crop_main'
wheat_fr_AVS_crop_main.save()



# Soy

soy_ch_AVS_elec_main,soy_ch_ref,soy_ch_AVS_crop_main = prepare_act_agri(soy_ch)

soy_ch_AVS_elec_main["database"] = 'foregroundAVS'
soy_ch_AVS_elec_main["name"] = 'soy_ch_AVS_elec_main'
soy_ch_AVS_elec_main["code"] = 'soy_ch_AVS_elec_main'
soy_ch_AVS_elec_main.save()

soy_ch_ref["database"] = 'foregroundAVS'
soy_ch_ref["name"] = 'soy_ch_ref'
soy_ch_ref["code"] = 'soy_ch_ref'
soy_ch_ref.save()

soy_ch_AVS_crop_main["database"] = 'foregroundAVS'
soy_ch_AVS_crop_main["name"] = 'soy_ch_AVS_crop_main'
soy_ch_AVS_crop_main["code"] = 'soy_ch_AVS_crop_main'
soy_ch_AVS_crop_main.save()



# Alfalfa

alfalfa_ch_AVS_elec_main,alfalfa_ch_ref,alfalfa_ch_AVS_crop_main = prepare_act_agri(alfalfa_ch)

alfalfa_ch_AVS_elec_main["database"] = 'foregroundAVS'
alfalfa_ch_AVS_elec_main["name"] = 'alfalfa_ch_AVS_elec_main'
alfalfa_ch_AVS_elec_main["code"] = 'alfalfa_ch_AVS_elec_main'
alfalfa_ch_AVS_elec_main.save()

alfalfa_ch_ref["database"] = 'foregroundAVS'
alfalfa_ch_ref["name"] = 'alfalfa_ch_ref'
alfalfa_ch_ref["code"] = 'alfalfa_ch_ref'
alfalfa_ch_ref.save()

alfalfa_ch_AVS_crop_main["database"] = 'foregroundAVS'
alfalfa_ch_AVS_crop_main["name"] = 'alfalfa_ch_AVS_crop_main'
alfalfa_ch_AVS_crop_main["code"] = 'alfalfa_ch_AVS_crop_main'
alfalfa_ch_AVS_crop_main.save()


# Our AVS now needs to consider the associated production of crop as a coproduct.
# 1 virtual unit of the three crops productions. This is the same as putting all the inputs and emissions of the crops to the AVS, and substuting with the crop output.


AVS_elec_main.new_exchange(amount=1, input=wheat_fr_AVS_elec_main, type="technosphere").save()
AVS_elec_main.new_exchange(amount=1, input=alfalfa_ch_AVS_elec_main, type="technosphere").save()
AVS_elec_main.new_exchange(amount=1, input=soy_ch_AVS_elec_main, type="technosphere").save()



AVS_crop_main.new_exchange(amount=1, input=wheat_fr_AVS_crop_main, type="technosphere").save()
AVS_crop_main.new_exchange(amount=1, input=alfalfa_ch_AVS_crop_main, type="technosphere").save()
AVS_crop_main.new_exchange(amount=1, input=soy_ch_AVS_crop_main, type="technosphere").save()



# for exc in list(AVS_elec_main.exchanges()):
#     print(exc)
    
   
    


""" ilUC """

# We also create virtual activities for iLUC. One each for AVS, Pvref and Wheatref







# First, add an exhange of carbon  dioxide  to the virtual carbon accumulation act in the foreground.



# Exchange: 1.42397953472488 kilogram 'Carbon dioxide, in air' (kilogram, None, ('natural resource', 'in air')) to 'wheat grain production' (kilogram, FR, None)>
# Carbon dioxide, in air
# cc6a1abb-b123-4ca6-8f16-38209df609be
Carbon_dioxide_to_soil_biomass_stock = Biosphere.get('375bc95e-6596-4aa1-9716-80ff51b9da77')

c_soil_accu.new_exchange(amount=44/12, input=Carbon_dioxide_to_soil_biomass_stock, type="biosphere").save()
c_soil_accu.save()


# Adds the virtual carbon accumulation activity to the different systems

wheat_fr_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
soy_ch_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
alfalfa_ch_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()

PV_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()


AVS_elec_main.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
AVS_crop_main.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()

# The accelerated impact category has a biosphere output of a new emission flow "iluc CO2eq" which is already characterized.
# to avoid double accounting, we create this new susbstance in a new biosphere database. 
# We also create a new lcia method which is Recipe GWP100 modified to include a cf of 1 kg CO2 eq for our new flow.
# Thus iLUC emissions are only characterized into GWP impact as we don't have access to the full model.

# ('Carbon dioxide, from soil or biomass stock' (kilogram, None, ('air', 'low population density, long-term')),


# https://stackoverflow.com/questions/44551595/create-a-new-method-in-brightway-2
additional_biosphere = bd.Database('additional_biosphere')

additional_biosphere.write({('additional_biosphere', "iLUCspecificCO2eq"): {
    'name': 'iLUCspecificCO2eq',
    'unit': 'kg CO2 eq',  
    'type': 'emission'
}})



    

# Create a specific method which extends the original RECIPE GWP 100.
recipegwp100 = ('ReCiPe 2016 v1.03, midpoint (H)',
                'climate change', 'global warming potential (GWP100)')

recipegwp100_data = bd.Method(recipegwp100).load() 

# We add the CF for our iluc specific GW emission
LCIAdatailUC = recipegwp100_data+[(additional_biosphere.get("iLUCspecificCO2eq").id, 1.0)]

method_key = ('ReCiPe 2016 v1.03, midpoint WITH ILUC (H)',
              'climate change',
              'global warming potential (GWP100) ILUC')

modifiedrecipewithiluc = bd.Method(method_key)
#modifiedrecipewithiluc.validate(LCIAdatailUC)
modifiedrecipewithiluc.register()
modifiedrecipewithiluc.write(LCIAdatailUC)
modifiedrecipewithiluc.load()


# We add this emission to our virtual act.
iLUCspecificCO2eq=additional_biosphere.get("iLUCspecificCO2eq")

iluc.new_exchange(
    amount=1, input=iLUCspecificCO2eq, type="biosphere").save()


# Add virtual exchanges of iluc to the luc activities 
LUCmarket_cropref.new_exchange(amount=1, input=iluc, type="technosphere").save()

LUCmarket_PVref.new_exchange(amount=1, input=iluc, type="technosphere").save()

LUCmarket_AVS.new_exchange(amount=1, input=iluc, type="technosphere").save()


#  Add eLUC market exchanges to the AVS, PV prod and Crop ref

wheat_fr_ref.new_exchange(amount=1, input=LUCmarket_cropref, type="technosphere").save()
soy_ch_ref.new_exchange(amount=1, input=LUCmarket_cropref, type="technosphere").save()
alfalfa_ch_ref.new_exchange(amount=1, input=LUCmarket_cropref, type="technosphere").save()

AVS_elec_main.new_exchange(amount=1, input=LUCmarket_AVS, type="technosphere").save()

AVS_crop_main.new_exchange(amount=1, input=LUCmarket_AVS, type="technosphere").save()

PV_ref.new_exchange(amount=1, input=LUCmarket_PVref,type="technosphere").save()


# AVS_crop_main substitutes either marginal or PV elec. To set up the database, we add an echnage of the 2 options.
# One of them will be set to 0 for each MC iteration.


elec_marginal_fr = Ecoinvent.get("a3b594fa27de840e85cb577a3d63d11a")

elec_marginal_fr_copy = elec_marginal_fr.copy()


elec_marginal_fr_copy["database"] = 'foregroundAVS'
elec_marginal_fr_copy["name"] = 'elec_marginal_fr_copy'
elec_marginal_fr_copy["code"] = 'elec_marginal_fr_copy'
elec_marginal_fr_copy.save()


AVS_crop_main.new_exchange(amount=-1, input=elec_marginal_fr_copy, type="technosphere").save()
AVS_crop_main.new_exchange(amount=-1, input=PV_ref, type="technosphere").save()

