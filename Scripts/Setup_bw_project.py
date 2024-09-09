# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:55:04 2024

@author: pierre.jouannais

Setup the brightway project.
Change the username and password for your ecoinvent account l23.
"""

from bw2data.parameters import *
import brightway2 as bw


import bw2data as bd
import bw2io as bi


from ecoinvent_interface import Settings, permanent_setting
permanent_setting("username", "xxx") # put your ecoinvent username instead of the xxx
permanent_setting("password", "xxx")  # put your ecoinvent password instead of the xxx


bd.projects.set_current("AVS") # your project

#This downloads and loads ecoinvent and the corresponding biosphere + the methods
bi.import_ecoinvent_release("3.10", "consequential") # ask for the version 


Ecoinvent = bw.Database('ecoinvent-3.10-consequential')


