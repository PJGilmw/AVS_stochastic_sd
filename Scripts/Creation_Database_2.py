# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:58:27 2024

@author: pierre.jouannais

Script that finishes setting up the foreground database.
To be executed as a whole.





Possible improvements:
Make copies of the wafer production so that its parameterization only 
affets the foreground activity. In this version, the modification of 
the wafer production is happening in the background but it has almost no effect on the different FUs.
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
import scipy as sc
from itertools import *

import functools
import re
import types


bd.projects.set_current('AVS')


foregroundAVS = bd.Database('foregroundAVS')

Ecoinvent = bd.Database('ecoinvent-3.10-consequential')

biosphere = bd.Database('ecoinvent-3.10-biosphere')




# We start by making a copy of 'photovoltaic slanted-roof installation, 3kWp, single-Si, panel, mounted, on roof'



#'photovoltaic slanted-roof installation, 3kWp, single-Si, panel, mounted, on roof' (unit, RoW, None),
 #'97372df133161bc1c9a89760d490fa0f')
photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof = Ecoinvent.get("97372df133161bc1c9a89760d490fa0f")


photovoltaicmono_installation_perm2panel_AVS= photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof .copy()

photovoltaicmono_installation_perm2panel_AVS["database"]='foregroundAVS'
photovoltaicmono_installation_perm2panel_AVS["name"] = 'photovoltaicmono_installation_perm2panel_AVS'
photovoltaicmono_installation_perm2panel_AVS["code"] = 'photovoltaicmono_installation_perm2panel_AVS'

photovoltaicmono_installation_perm2panel_AVS.save()





for exc in list(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof.exchanges()):
    print(exc)
    
# Panel PV normal

photovoltaicmono_installation_perm2panel_PV= photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof.copy()
photovoltaicmono_installation_perm2panel_PV["database"]='foregroundAVS'
photovoltaicmono_installation_perm2panel_PV["name"] = 'photovoltaicmono_installation_perm2panel_PV'
photovoltaicmono_installation_perm2panel_PV["code"] = 'photovoltaicmono_installation_perm2panel_PV'
photovoltaicmono_installation_perm2panel_PV.save()


    
# Explore

for exc in list(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            print(input_name,input_loc, exc["amount"] ,exc["unit"])
            
            

#we remove the mounting systems in both PV and AVS installation


for exc in list(photovoltaicmono_installation_perm2panel_PV.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":

            input_= Ecoinvent.get(exc["input"][1])
            input_name = input_["name"]

            if "market for photovoltaic mounting system" in input_name:
                
            # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
            # exc.save()
                exc.delete()
                
                
                
for exc in list(photovoltaicmono_installation_perm2panel_AVS.exchanges()):

        if exc["type"]=="technosphere":

            input_= Ecoinvent.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            if "market for photovoltaic mounting system" in input_name:
                
            # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
            # exc.save()
                exc.delete()
                                
          
#ok

#rescaled all electricity inputs to 1 square meter


for exc in list(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        print(exc)
        print(exc["name"])
        if "photovoltaic panel, single-Si wafer" in exc["name"]:
            original_surface_panel = exc["amount"]

for exc in list(photovoltaicmono_installation_perm2panel_AVS.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":

            input_= Ecoinvent.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            if "electricity" in input_name or "market for photovoltaic panel" in input_name:
              print(input_name)  
              exc_former = exc["amount"]

              exc["amount"] = exc_former/original_surface_panel #22.07187 in ecoinent 3.9
              exc.save()
              

for exc in list(photovoltaicmono_installation_perm2panel_PV.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":

            input_= Ecoinvent.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            if "electricity" in input_name or "market for photovoltaic panel" in input_name:
                
              exc_former = exc["amount"]

              exc["amount"] = exc_former/original_surface_panel #22.07187 in ecoinent 3.9
              exc.save()
              


        
# Now we will create modifed version of the inverter, the electic installation and the panel


"""Electric installation"""



#('photovoltaics, electric installation for 570kWp module, open ground' (unit, GLO, None),'772b78c3f69621d3c9927903c8b56b37')    
photovoltaics_electric_installation_570kWpmodule = Ecoinvent.get("772b78c3f69621d3c9927903c8b56b37")

#electric installation 
list_inputs_electric_instal_id = []
list_inputs_electric_instal_weights = []

for exc in list(photovoltaics_electric_installation_570kWpmodule.exchanges()):
  
        input_= Ecoinvent.get(exc["input"][1])
        input_name = input_["name"]
        input_loc = input_["location"]
        
        print(input_name,input_loc, exc["amount"],exc["unit"])
        
        list_inputs_electric_instal_id.append(input_.id)
        if exc["amount"]>0 and exc["unit"]=="kilogram":
            list_inputs_electric_instal_weights.append(exc["amount"])
    
list_inputs_electric_instal_weights
sum(list_inputs_electric_instal_weights)

# 2308 kg of electric installa for 570kwp . 2308/570 =  4 kg/kwp
electricpvinstallation_kg= photovoltaics_electric_installation_570kWpmodule.copy()

electricpvinstallation_kg["database"]='foregroundAVS'
electricpvinstallation_kg["name"] = 'electricpvinstallation_kg'
electricpvinstallation_kg["code"] = 'electricpvinstallation_kg'
electricpvinstallation_kg["unit"] = 'kg'

electricpvinstallation_kg.save()

# It is 1 kg of electrical installation. 

for exc in list(electricpvinstallation_kg.exchanges()):
    if exc["type"]!="production":
        exc_former = exc["amount"]
        exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        exc.save()
  
electricpvinstallation_kg.save()




# We replace the input of installation by our new activity. 
# So far we put 1, but the amount will be parameterized.

#first delete original
for exc in list(photovoltaicmono_installation_perm2panel_PV.exchanges()):
        
        if exc["type"]=="technosphere":

            input_= Ecoinvent.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            if "electric installation" in input_name:
                
              exc.delete()
         
for exc in list(photovoltaicmono_installation_perm2panel_AVS.exchanges()):

        if exc["type"]=="technosphere":

            input_= Ecoinvent.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            if "electric installation" in input_name:
                
              exc.delete()     

# Now add exchange of the reolacement activity
photovoltaicmono_installation_perm2panel_AVS.new_exchange(amount=1, input=electricpvinstallation_kg , type="technosphere",unit="kilogram").save()
photovoltaicmono_installation_perm2panel_PV.new_exchange(amount=1, input=electricpvinstallation_kg , type="technosphere",unit="kilogram").save()



 
    

"""Inverter"""




#Remove inverter

for exc in list(photovoltaicmono_installation_perm2panel_AVS.exchanges()):

        if exc["type"]=="technosphere":
            print("oook")
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
                
            input_name = input_["name"]
            input_loc = input_["location"]
            if "market for inverter" in input_name:
                
              exc.delete()
              
              
              
for exc in list(photovoltaicmono_installation_perm2panel_PV.exchanges()):

        if exc["type"]=="technosphere":
            print("oook")
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
                
            input_name = input_["name"]
            input_loc = input_["location"]
            if "market for inverter" in input_name:
                
              exc.delete()
                            

          

# No need to interpolate cause the VERY minimum installed power will already be around 100 kwp
# So let's just take the 500 kwp modified inverter 


# 'market for inverter, 500kW' (unit, GLO, None)
market_for_inverter_500kW = Ecoinvent.get('47c33fcb8cde05ad84b16d60e575c566')

#'inverter production, 500kW' (unit, RoW, None)
inverter_500_row =Ecoinvent.get('f2e81592e5571fa313736f76c6153f13')

#'inverter production, 500kW' (unit, RER, None)

inverter_500_rer =Ecoinvent.get('59ccf4ea27a082ed992a2d34adca243e')



market_for_inverter_500kW_kg=market_for_inverter_500kW.copy()
market_for_inverter_500kW_kg["database"]="foregroundAVS"
market_for_inverter_500kW_kg["name"] = "market_for_inverter_500kW_kg"
market_for_inverter_500kW_kg["code"] = "market_for_inverter_500kW_kg"
market_for_inverter_500kW_kg["unit"] = 'kilogram'

market_for_inverter_500kW_kg.save()



# This was done to rescale to 1 kg of inverter

list_inputs_inverter_id = []
list_inputs_inverter_weights = []
df_=pd.DataFrame({'input':[], 
        'amount':[], 
        'unit':[] } )

# Calculate total weightt


for exc in list(inverter_500_rer.exchanges()):
  
        input_= Ecoinvent.get(exc["input"][1])
        input_name = input_["name"]
        input_loc = input_["location"]
        
        print(input_name,input_loc, exc["amount"],exc["unit"])
        df_.loc[len(df_.index)] = [ input_name, exc["amount"], exc["unit"]]  

        list_inputs_inverter_id.append(input_.id)
        if exc["amount"]>0 and exc["type"]!="production" and exc["unit"]=="kilogram":
            list_inputs_inverter_weights.append(exc["amount"])

weigth_500kWrer= df_[(df_.unit == 'kilogram')&(df_.amount > 0)].amount.drop_duplicates().sum()

weigth_500kWrer/500  # 5.97 kg inverter per kWP

inverter_500_rer_kg=inverter_500_rer.copy()
inverter_500_rer_kg["database"]="foregroundAVS"
inverter_500_rer_kg["name"] = "inverter_500_rer_kg"
inverter_500_rer_kg["code"] = "inverter_500_rer_kg"
inverter_500_rer_kg["unit"] = 'kilogram'

inverter_500_rer_kg.save()



for exc in list(inverter_500_rer_kg.exchanges()):
    if exc["type"]!="production":
        exc_former = exc["amount"]
        exc["amount"] = exc_former/weigth_500kWrer ####
        exc.save()



    

# RoW

inverter_500_row_kg=inverter_500_row.copy()
inverter_500_row_kg["database"]="foregroundAVS"
inverter_500_row_kg["name"] = "inverter_500_row_kg"
inverter_500_row_kg["code"] = "inverter_500_row_kg"
inverter_500_row_kg["unit"] = 'kilogram'



list_inputs_inverter_id = []
list_inputs_inverter_weights = []
df_=pd.DataFrame({'input':[], 
        'amount':[], 
        'unit':[] } )
# Calculate total weight
for exc in list(inverter_500_row.exchanges()):
  
        input_= Ecoinvent.get(exc["input"][1])
        input_name = input_["name"]
        input_loc = input_["location"]
        
        print(input_name,input_loc, exc["amount"],exc["unit"])
        df_.loc[len(df_.index)] = [ input_name, exc["amount"], exc["unit"]]  

        list_inputs_inverter_id.append(input_.id)
        if exc["amount"]>0 and exc["unit"]=="kilogram":
            list_inputs_inverter_weights.append(exc["amount"])

print('Weight of the inverter : %.0f kg'%df_[(df_.unit == 'kilogram')&(df_.amount > 0)].amount.drop_duplicates().sum())
weigth_500kWrow= df_[(df_.unit == 'kilogram')&(df_.amount > 0)].amount.drop_duplicates().sum()


for exc in list(inverter_500_row_kg.exchanges()):
    if exc["type"]!="production":        
        exc_former = exc["amount"]
        exc["amount"] = exc_former/weigth_500kWrow
        exc.save()
        
        
    
        
# Now we add the new inverters to the market


for exc in list(market_for_inverter_500kW_kg.exchanges()):
        if exc["type"]!="production"    :
    
            input_= Ecoinvent.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            if input_loc =="RER":
                exc_former = exc["amount"]
                
                market_for_inverter_500kW_kg.new_exchange(amount=exc_former,
                                                             input=inverter_500_rer_kg, 
                                                             type="technosphere",
                                                             unit="kilogram").save()
                exc.delete()
            else:
                exc_former = exc["amount"]
                
                market_for_inverter_500kW_kg.new_exchange(amount=exc_former,
                                                             input=inverter_500_row_kg, 
                                                             type="technosphere",
                                                             unit="kilogram").save()
                exc.delete()
market_for_inverter_500kW_kg.save()               

        
        
# Now add exchange new inverter of the reolacement activity
photovoltaicmono_installation_perm2panel_AVS.new_exchange(amount=1, input=market_for_inverter_500kW_kg , type="technosphere",unit="kilogram").save()
photovoltaicmono_installation_perm2panel_PV.new_exchange(amount=1, input=market_for_inverter_500kW_kg, type="technosphere",unit="kilogram").save()
           



""" Modification of the PV panel production"""




# 'market for photovoltaic panel, single-Si wafer' (square meter, GLO, None)
marketforphotovoltaicpanelsingle_Si_wafer = Ecoinvent.get('095784514d394dfeecf399a047d006c8')
   
            


for exc in list(marketforphotovoltaicpanelsingle_Si_wafer.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            print(input_name,input_loc, exc["amount"] ,exc["unit"],input_["code"])





listin=[(act,act["code"]) for act in Ecoinvent if 'photovoltaicpanelsingle_Si_prod_row_modif'in act['name'] ]
listin

""""""
#'photovoltaic panel production, single-Si wafer' (square meter, RER, None)
photovoltaicpanelsingle_Si_prod_rer =Ecoinvent.get('7ef0c463fcc7c391e9676e53743b977f')

# 'photovoltaic panel production, single-Si wafer' (square meter, RoW, None)
photovoltaicpanelsingle_Si_prod_row =Ecoinvent.get('9e0b81cf2d44559f13849f03e7b3344d')
""""""



# Wafer prod


waferprod=[(act,act["code"],act.id) for act in Ecoinvent if 'photovoltaic cell production, single-Si wafer'in act['name'] ]
waferprod





wafer_row = Ecoinvent.get("69e0ae62d7d13ca6270e9634a0c40374")
wafer_rer = Ecoinvent.get("4ced0dbf5e0dbf56245b175b8171a6fb")



for exc in list(wafer_row.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            print(input_name,input_loc, exc["amount"] ,exc["unit"],input_["code"],input_.id)

for exc in list(wafer_rer.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            print(input_name,input_loc, exc["amount"] ,exc["unit"],input_["code"],input_.id)





# Solar grade

actsolargrade=[(act,act["code"]) for act in Ecoinvent if 'silicon production, solar grade, modified Siemens process'in act['name'] ]
actsolargrade


# The only two activities consituting the market for solar grade silicon (different from attributional)
siliconproductionsolar_grade_row=Ecoinvent.get("2843a8c71e81a4134b27c8918f018372")
siliconproductionsolar_grade_rer=Ecoinvent.get("7f5d2c4c04f1e4d8733f9ba103f08822")

# RER

for exc in list(siliconproductionsolar_grade_rer.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            print(input_name,input_loc, exc["amount"] ,exc["unit"],input_["code"])

# In a fix dp
# we remove the hydro electricity ROw which has nothing to do here
electricity_hydro = Ecoinvent.get("0dc865061225e93277c254656dfc1260")
#the input that must be set to 0 is at:
(electricity_hydro.id,siliconproductionsolar_grade_rer.id)
 #(23304, 14265)
 
 #Parameterized dp
 
# the RER electicity input is parameterized with solargrade_electric_intensity
electricity_RER_margetgroup= Ecoinvent.get("1e1014b29f9e44d8265fcd8d871c9cce")
#the input that must be parameterized;
(electricity_RER_margetgroup.id,siliconproductionsolar_grade_rer.id)
 #(14020, 14265)
    



# RoW

for exc in list(siliconproductionsolar_grade_row.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            print(input_name,input_loc, exc["amount"] ,exc["unit"],input_["code"])


# we remove the hydro electricity ROw which has nothing to do here
electricity_hydro = Ecoinvent.get("0dc865061225e93277c254656dfc1260")
#the input that must be set to 0 is at:
(electricity_hydro.id,siliconproductionsolar_grade_row.id)
 #(23304, 15392)
 
 #Parameterized dp
 
 
# Calculate total elec
total_elec=[]
elec_id=[]
for exc in list(siliconproductionsolar_grade_row.exchanges()):
        # exc_former = exc["amount"]
        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
        # exc.save()
        if exc["type"]=="technosphere":
            if exc["input"][0] =="ecoinvent-3.10-consequential":
                input_= Ecoinvent.get(exc["input"][1])
            else:    
                input_= foregroundAVS.get(exc["input"][1])
            input_name = input_["name"]
            input_loc = input_["location"]
            
            if "electricity" in input_name:
                total_elec.append(exc["amount"])
                elec_id.append(input_.id)

 
 
# We keep the hydro prod here
 
def f_elec_intensity_solargraderow(init,
                                   elec_intensity):
    new_value = init*elec_intensity/110 # 110 kWh in total in the original act

    return new_value




mountingsystem=[(act,act["code"]) for act in Ecoinvent if 'photovoltaic mounting system'in act['name'] ]
mountingsystem

# 'photovoltaic mounting system production, for 570kWp open ground module' (square meter, GLO, None)

mountingsystem = Ecoinvent.get('a876f2ce9a969fbc7bbe76f3b40a1901')

for exc in list(mountingsystem.exchanges()):
    print(exc)
    
#3.98+7.25+  0.25
            
mountingsystem_modif_PV = mountingsystem.copy()   

mountingsystem_modif_PV["name"]="mountingsystem_modif_PV"
mountingsystem_modif_PV["database"] = "foregroundAVS"
mountingsystem_modif_PV["code"]="mountingsystem_modif_PV"
mountingsystem_modif_PV.save()


            
mountingsystem_modif_AVS = mountingsystem.copy()   

mountingsystem_modif_AVS["name"]="mountingsystem_modif_AVS"
mountingsystem_modif_AVS["database"] = "foregroundAVS"
mountingsystem_modif_AVS["code"]="mountingsystem_modif_AVS"
mountingsystem_modif_AVS.save()



# Calculated from the ratio occupation/tranfo
lifetimemountingstrucutre=141/4.7  # 30 years
            
# Delete occupation and transfo

for exc in list(mountingsystem_modif_AVS.exchanges()):
    if exc["type"]== "biosphere":
        
        exc.delete()
mountingsystem_modif_AVS.save()        

for exc in list(mountingsystem_modif_PV.exchanges()):
    if exc["type"]== "biosphere":
        
        exc.delete()
mountingsystem_modif_PV.save()        



# Add inputs of panel and mounting structure to the AVS and PV production

# Mounting structure

AVS_elec_main= foregroundAVS.get("AVS_elec_main")
AVS_crop_main= foregroundAVS.get("AVS_crop_main")

PV_ref= foregroundAVS.get("PV_ref")

AVS_elec_main.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()

AVS_crop_main.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()

PV_ref.new_exchange(amount=1, input=mountingsystem_modif_PV, type="technosphere").save()



# Panel instal


AVS_elec_main.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_AVS, type="technosphere").save()
AVS_crop_main.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_AVS, type="technosphere").save()


PV_ref.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_PV, type="technosphere").save()

AVS_elec_main.save()
AVS_crop_main.save()

PV_ref.save()

methodimp = ('ReCiPe 2016 v1.03, midpoint (H)',
                'climate change', 'global warming potential (GWP100)')


# Add occupation of industrial area for PV ref.
# AVS has kept the same occupation as the associated crop.
occupation_industrial_code = [act["code"] for act in biosphere if 'Occupation, industrial area' in act["name"]][0]
occupation_industrial=biosphere.get(occupation_industrial_code)

PV_ref.new_exchange(amount=10000, input=occupation_industrial, type="biosphere").save()

# elec prod marginal france
elec_marginal_fr = Ecoinvent.get("a3b594fa27de840e85cb577a3d63d11a")






