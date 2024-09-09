# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:38:14 2024

@author: pierre.jouannais



This script simply collects the different activities from the foregound and background databases and assign them to variables.
These variables are then called in other scripts.

"""




import bw2data as bd



bd.projects.set_current('AVS')


Ecoinvent = bd.Database('ecoinvent-3.10-consequential')

Biosphere = bd.Database('ecoinvent-3.10-biosphere')



foregroundAVS = bd.Database('foregroundAVS')

additional_biosphere = bd.Database('additional_biosphere')



# FG
AVS_elec_main = foregroundAVS.get("AVS_elec_main")
AVS_crop_main = foregroundAVS.get("AVS_crop_main")

PV_ref  = foregroundAVS.get("PV_ref")


LUCmarket_AVS = foregroundAVS.get("LUCmarket_AVS")
LUCmarket_PVref= foregroundAVS.get("LUCmarket_PVref")
LUCmarket_cropref= foregroundAVS.get("LUCmarket_cropref")
iLUCspecificCO2eq= additional_biosphere.get("iLUCspecificCO2eq")
iluc=foregroundAVS.get("iluc")

wheat_fr_ref = foregroundAVS.get("wheat_fr_ref")
soy_ch_ref = foregroundAVS.get("soy_ch_ref")
alfalfa_ch_ref = foregroundAVS.get("alfalfa_ch_ref")


# Get the outputs of these reference crop activities
output_wheat = [exc["amount"] for exc in list(wheat_fr_ref.exchanges()) if exc["type"]=="production"][0]
output_soy = [exc["amount"] for exc in list(soy_ch_ref.exchanges()) if exc["type"]=="production"][0]
output_alfalfa = [exc["amount"] for exc in list(alfalfa_ch_ref.exchanges()) if exc["type"]=="production"][0]





wheat_fr_AVS_crop_main = foregroundAVS.get("wheat_fr_AVS_crop_main")
soy_ch_AVS_crop_main = foregroundAVS.get("soy_ch_AVS_crop_main")
alfalfa_ch_AVS_crop_main = foregroundAVS.get("alfalfa_ch_AVS_crop_main")

wheat_fr_AVS_elec_main = foregroundAVS.get("wheat_fr_AVS_elec_main")
soy_ch_AVS_elec_main = foregroundAVS.get("soy_ch_AVS_elec_main")
alfalfa_ch_AVS_elec_main = foregroundAVS.get("alfalfa_ch_AVS_elec_main")


c_soil_accu = foregroundAVS.get("c_soil_accu")

# Market elec

elec_marginal_fr = Ecoinvent.get("a3b594fa27de840e85cb577a3d63d11a")
elec_marginal_fr_copy = foregroundAVS.get("elec_marginal_fr_copy")


# Inverter



mark_inv_500kW_kg = foregroundAVS.get("market_for_inverter_500kW_kg")

# Install 1 m2 PV
pv_insta_PV = foregroundAVS.get("photovoltaicmono_installation_perm2panel_PV")
pv_insta_AVS = foregroundAVS.get("photovoltaicmono_installation_perm2panel_AVS")




# Electric installation
elecinstakg =  foregroundAVS.get('electricpvinstallation_kg')


# Input of aluminium in the PV panel productions


#'photovoltaic panel production, single-Si wafer' (square meter, RER, None)
pvpanel_prod_rer = Ecoinvent.get('7ef0c463fcc7c391e9676e53743b977f')

# 'photovoltaic panel production, single-Si wafer' (square meter, RoW, None)

pvpanel_prod_row = Ecoinvent.get('9e0b81cf2d44559f13849f03e7b3344d')


#'market for aluminium alloy, AlMg3' (kilogram, GLO, None)
# Same for RoW Panel
aluminium_panel=Ecoinvent.get("8a81f71bde65e4274ff4407cdf0c6320")




wafer_row = Ecoinvent.get("69e0ae62d7d13ca6270e9634a0c40374")
wafer_rer = Ecoinvent.get("4ced0dbf5e0dbf56245b175b8171a6fb")


# Electricity into wafer
# Row
elec_wafer_nz = Ecoinvent.get("8f26c425a60628f008c1ebe4492a07f0")
elec_wafer_rla = Ecoinvent.get("2264d5f7468cf8b715bbb6ed97de6a73")
elec_wafer_raf = Ecoinvent.get("d51b43544c5f9476ab14145734d4bd2a")
elec_wafer_au = Ecoinvent.get("a48e9862afad47e1b51e47b63c7f4496")
elec_wafer_ci = Ecoinvent.get("547f5f74e17e3740124326a21008ec8e")
elec_wafer_rna = Ecoinvent.get("eb8cc93eaaa1801a14edca17a24f3995")
elec_wafer_ras = Ecoinvent.get("25d7147cb58dcd7f7d368a58d60b9eb5")


# RER
elec_wafer_rer = Ecoinvent.get("b7e29adbe4a6db565509fd0059b1d4a6")



# Silicon prod

si_sg_row=Ecoinvent.get("2843a8c71e81a4134b27c8918f018372")
si_sg_rer=Ecoinvent.get("7f5d2c4c04f1e4d8733f9ba103f08822")

elec_sili_raf = Ecoinvent.get("23198afc58716e2f16033b5e83b2e60a")
elec_sili_au = Ecoinvent.get("af26f5d13d727349e5c1837f8c202420")
elec_sili_ci = Ecoinvent.get("b8c8ace7fa5c0d619714bdbb57a7efef")
elec_sili_nz = Ecoinvent.get("9a5c551cc6e7d25cce868e3a26eeb96a")
elec_sili_ras = Ecoinvent.get("356270fb85780d1f7529ddedf445ac11")
elec_sili_rna = Ecoinvent.get("eb7dbda38cbb4371f58f1fbc477b0293")
elec_sili_rla = Ecoinvent.get("8d93fc8256c8af3f210fc0ebd6f2cc88")


#siliconproductionsolar_grade_row.id
#Applied to:
#[9474, 9516, 9648, 10752, 16409, 17973, 18732, 23304]


# In a fix dp
# we remove the hydro electricity ROw which has nothing to do here
electricity_hydro = Ecoinvent.get("0dc865061225e93277c254656dfc1260")


# the RER electicity input is parameterized with solargrade_electric_intensity
electricity_RER_margetgroup= Ecoinvent.get("1e1014b29f9e44d8265fcd8d871c9cce")
#the input that must be parameterized;






mount_system_AVS = foregroundAVS.get("mountingsystem_modif_AVS")

mount_system_PV = foregroundAVS.get("mountingsystem_modif_PV")



concrete_mount=Ecoinvent.get("bc8c7359f05890c22e37ab303bdd987a")


concrete_mount_waste_1=Ecoinvent.get("e399ae8908f7a23700aad8c833ece1ca")

concrete_mount_waste_2=Ecoinvent.get("5df13a7c86091d82d261b14090992eae")

concrete_mount_waste_3=Ecoinvent.get("d751bebe3aa8f622fc868d3b6a4ea6c4")


aluminium_mount =Ecoinvent.get("b2a028e18853749b07a3631ef6eccd96") 

alu_extru_mount =Ecoinvent.get("0d52d26769eb8011ebc49e09efb7259a") 

alu_mount_scrap_1 =Ecoinvent.get("238a2639dfe3ecbcd10a54426cd01094") 

alu_mount_scrap_2 =Ecoinvent.get("b487525640db147e00a2ece275262d59") 

alu_mount_scrap_3 =Ecoinvent.get("b7acf0e07b901a83ab9071d87e881af2") 


reinf_steel_mount =Ecoinvent.get("1737592f8b159167b376ff1ffe485e7e") 
chrom_steel_mount =Ecoinvent.get("ecc7db7759925cf080ec73926ca4470e") 

steel_rolling_mount =Ecoinvent.get("a19e6d008c28760e4872a5e7e001be0f") 


wire_mount = Ecoinvent.get("bc2fb8454c1dcf38750e1272ee1aae10")
steel_mount_scrap_1= Ecoinvent.get("d8deb03d160b32aff1962dd626878c1e")
steel_mount_scrap_2= Ecoinvent.get("b9e7ab6f52bf1a1c2079b7fd6da7139e")
steel_mount_scrap_3= Ecoinvent.get("f95f81d06894ce8cd3992b22592a7a2e")

# Corrugated carb box

cor_box_mount_1 = Ecoinvent.get("3c297273fd9dd888ca5f562b60633b1c")
cor_box_mount_2 = Ecoinvent.get("88b4c06b619444f94dbadd73ba25a99a")
cor_box_mount_3 = Ecoinvent.get("bc6d87ea5c33af362a873002e89d0a67")
cor_box_mount_4 = Ecoinvent.get("da47edf1c623e16e14412cf859e038ef")

cor_box_mount_waste_1 = Ecoinvent.get("0fbef1744ad94621c41e6a29eff631e6")
cor_box_mount_waste_2 = Ecoinvent.get("07135843544f495acdec3805593597cd")
cor_box_mount_waste_3 = Ecoinvent.get("5adbcfff2d45da42b0c894714357be1d")


poly_mount_1 = Ecoinvent.get("fe37424fcb39751b8436bea56516d595")
poly_mount_2 = Ecoinvent.get("ede9bf12aa723d584c162ce6a6805974")

poly_mount_waste_1 = Ecoinvent.get("47e42f31205177ca7ccd475552344b26")
poly_mount_waste_2 = Ecoinvent.get("429bdbc2829990503a76dcae0b79e98d")
poly_mount_waste_3 = Ecoinvent.get("96206cf1b984ab9666c6c84ad37a9f62")
poly_mount_waste_4 = Ecoinvent.get("5db45c95744227b5dbf5368b217666d7")
poly_mount_waste_5 = Ecoinvent.get("4ab43cbf5ef6529a43b65011a45b4352")
poly_mount_waste_6 = Ecoinvent.get("ada15830ef22c5f76c41c633cafe81aa")

zinc_coat_mount_1 = Ecoinvent.get("76c32858d64d75927d5a94a5a8683571") 
zinc_coat_mount_2 = Ecoinvent.get("00d441215fd227f4563ff6c2f4ae3f02") 


# Alfalfa input and ouput



# Exchange: 6.6662e-05 hectare 'fertilising, by broadcaster' (hectare, CH, None) to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None)>
# fertilising, by broadcaster
# 302a1e9ca5f7b39a5c6f4a7e27bd56c0
fert_broadcaster_ch=Ecoinvent.get("302a1e9ca5f7b39a5c6f4a7e27bd56c0")



# Exchange: 0.0042024 kilogram 'market for inorganic phosphorus fertiliser, as P2O5' (kilogram, CH, None) to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None)>
# market for inorganic phosphorus fertiliser, as P2O5
# adf4a377ba3fca72cb0d2090757a0bb1

ino_P205_ch=Ecoinvent.get("adf4a377ba3fca72cb0d2090757a0bb1")


# Exchange: 0.0013332 cubic meter 'liquid manure spreading, by vacuum tanker' (cubic meter, CH, None) to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None)>
# liquid manure spreading, by vacuum tanker
# f8c1374cd1f5f1f800a873ee4feb590e

liquidmanure_spr_ch=Ecoinvent.get("f8c1374cd1f5f1f800a873ee4feb590e")





# Exchange: 0.0084048 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None) to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None)>
# market for packaging, for fertilisers
# a9254f81cc8ecaeb85ab44c32b2e02be
packaging_fert_glo =Ecoinvent.get("a9254f81cc8ecaeb85ab44c32b2e02be")

# Exchange: 0.66662 kilogram 'solid manure loading and spreading, by hydraulic loader and spreader' (kilogram, CH, None) to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None)>
# solid manure loading and spreading, by hydraulic loader and spreader
# d9355b12099cf24dd94f024cfca32d35
solidmanure_spreading_ch = Ecoinvent.get("d9355b12099cf24dd94f024cfca32d35")



# Machinery
fodder_loading_ch =  Ecoinvent.get("10f89cf01395e21928b4137637b523a8")
rotary_mower_ch =  Ecoinvent.get("16735e3c568e791b8074941193b3ba12")
sowing_ch =  Ecoinvent.get("324c1476b808b7149f0ed9d7c8f5afed")
tillage_rotary_spring_tine_ch =  Ecoinvent.get("d408ce3256fb75715ddad41fcc92cebc")
tillage_ploughing_ch = Ecoinvent.get("457c5b6648cad5aecc781c9f5d5eca55")
tillage_rolling_ch = Ecoinvent.get("11374fa3faed9fe1cbab183ece4fa00e")





combine_harvesting_ch = Ecoinvent.get("967325bd96bf98cd0d180527060236f9")



"""Soy_ch"""


# for exc in list(soy_ch_AVS.exchanges()):
#     if exc["type"]=="biosphere" and "Water" in exc["name"]:
#         print(exc["name"])
        


# for exc in list(alfalfa_ch_AVS.exchanges()):
#     if exc["type"]=="biosphere" and "Water" in exc["name"]:
#         print(exc["name"])
        
        
# for exc in list(wheat_fr_AVS.exchanges()):
#     if exc["type"]=="biosphere"and "Water" in exc["name"]:
#         print(exc["name"])        
# Exchange: 0.00026044 hectare 'fertilising, by broadcaster' (hectare, CH, None) to 'soybean production' (kilogram, CH, None)>
# fertilising, by broadcaster
# 302a1e9ca5f7b39a5c6f4a7e27bd56c0
fert_broadcaster_ch=Ecoinvent.get("302a1e9ca5f7b39a5c6f4a7e27bd56c0")

#Exchange: 0.00026044 hectare 'green manure growing, Swiss integrated production, until January' (hectare, CH, None) to 'soybean production' (kilogram, CH, None)
#green manure growing, Swiss integrated production, until January
green_manure_ch =Ecoinvent.get("7dc3dcdf0989c1fe66d1cc94ac057726")


# Exchange: 0.00066337 kilogram 'nutrient supply from thomas meal' (kilogram, GLO, None) to 'soybean production' (kilogram, CH, None)>
# nutrient supply from thomas meal
# 550fc5c30f02c82d6a7f491a6cfbf4c4


# HERE NOT DEFINED FOR SOME REASON
nutrient_supply_thomas_meal_ch =Ecoinvent.get("550fc5c30f02c82d6a7f491a6cfbf4c4")


# Exchange: 0.0011157 cubic meter 'liquid manure spreading, by vacuum tanker' (cubic meter, CH, None) to 'soybean production' (kilogram, CH, None)>
# liquid manure spreading, by vacuum tanker
# f8c1374cd1f5f1f800a873ee4feb590e
liquidmanure_spr_ch =Ecoinvent.get("f8c1374cd1f5f1f800a873ee4feb590e")


# Exchange: 0.03943066 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None) to 'soybean production' (kilogram, CH, None)>
# market for packaging, for fertilisers
# a9254f81cc8ecaeb85ab44c32b2e02be
packaging_fert_glo =Ecoinvent.get("a9254f81cc8ecaeb85ab44c32b2e02be")


# Exchange: 0.009595625 kilogram 'market for phosphate rock, beneficiated' (kilogram, RER, None) to 'soybean production' (kilogram, CH, None)>
# market for phosphate rock, beneficiated
# 6ab3fd3b19187e0406419b03f3b88ff3
phosphate_rock_glo = Ecoinvent.get("6ab3fd3b19187e0406419b03f3b88ff3")


# Exchange: 0.0156317446180305 kilogram 'market for potassium chloride' (kilogram, RER, None) to 'soybean production' (kilogram, CH, None)>
# market for potassium chloride
# 3c83fdd5c69822caadf209bdbf8a50f3
potassium_chloride_rer = Ecoinvent.get("3c83fdd5c69822caadf209bdbf8a50f3")


# Exchange: 0.00119833261913457 kilogram 'market for potassium sulfate' (kilogram, RER, None) to 'soybean production' (kilogram, CH, None)>
# market for potassium sulfate
# 06f1d7f08f051e8e466af8fd4faaadd9
potassium_sulfate_rer = Ecoinvent.get("06f1d7f08f051e8e466af8fd4faaadd9")

# Exchange: 0.00100714285714286 kilogram 'market for single superphosphate' (kilogram, RER, None) to 'soybean production' (kilogram, CH, None)>
# market for single superphosphate
# 61f2763127305038f68860164831764d
single_superphosphate_rer = Ecoinvent.get("61f2763127305038f68860164831764d")



# Exchange: 0.072403 kilogram 'solid manure loading and spreading, by hydraulic loader and spreader' (kilogram, CH, None) to 'soybean production' (kilogram, CH, None)>
# solid manure loading and spreading, by hydraulic loader and spreader
# d9355b12099cf24dd94f024cfca32d35
solidmanure_spreading_ch = Ecoinvent.get("d9355b12099cf24dd94f024cfca32d35")

# Exchange: 0.0114058695652174 kilogram 'market for triple superphosphate' (kilogram, RER, None) to 'soybean production' (kilogram, CH, None)>
# market for triple superphosphate
# 10952b351896a1f0935e0dbd1fc98f49
triplesuperphosphate = Ecoinvent.get("10952b351896a1f0935e0dbd1fc98f49")

# Machinery
tillage_currying_weeder_ch =  Ecoinvent.get("9fdce9892f57840bd2fa96d92ab577e3")
tillage_rotary_spring_tine_ch =  Ecoinvent.get("d408ce3256fb75715ddad41fcc92cebc")
sowing_ch =  Ecoinvent.get("324c1476b808b7149f0ed9d7c8f5afed")

combine_harvesting_ch = Ecoinvent.get("967325bd96bf98cd0d180527060236f9")


"""Wheat fr"""


ammonium_nitrate = Ecoinvent.get("28acde92cd1aec3b5d5b8f58f50055ba")

ammonium_sulfate = Ecoinvent.get("3e5d910e09135106f6c712c538182b71")

urea = Ecoinvent.get("d729224f52da90228e5c4b7f864303b0")

# Exchange: 0.00051831 hectare 'market for fertilising, by broadcaster' (hectare, GLO, None) to 'wheat grain production' (kilogram, FR, None)>
# market for fertilising, by broadcaster
# 2433809f3b7d23df8625099f845e381
fert_broadcaster=Ecoinvent.get("2433809f3b7d23df8625099f845e3814")

# Exchange: 0.0104 kilogram 'market for inorganic phosphorus fertiliser, as P2O5' (kilogram, FR, None) to 'wheat grain production' (kilogram, FR, None)>


ino_P205_fr=Ecoinvent.get("12b793a4ed3da8e8e17209b4e788d9c0")


# Exchange: 3.406e-07 kilogram 'market for organophosphorus-compound, unspecified' (kilogram, GLO, None) to 'wheat grain production' (kilogram, FR, None)>
# market for organophosphorus-compound, unspecified
# cbabef8729a2ba6a1286742273fa3a6e
org_P205=Ecoinvent.get("cbabef8729a2ba6a1286742273fa3a6e")

# Exchange: 0.094817 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None) to 'wheat grain production' (kilogram, FR, None)>
# market for packaging, for fertilisers
# a9254f81cc8ecaeb85ab44c32b2e02be

packaging_fert = Ecoinvent.get("a9254f81cc8ecaeb85ab44c32b2e02be")


packaging_fert.id


# Exchange: 0.123183267669125 cubic meter 'market for irrigation' (cubic meter, FR, None) to 'wheat grain production' (kilogram, FR, None)>
# market for irrigation
# 432f8fdd083f6730c0f2ae6c8014656e
water_market_irrigation = Ecoinvent.get("432f8fdd083f6730c0f2ae6c8014656e")



# Machinery
tillage_rotary_harrow_glo =  Ecoinvent.get("cb6989c6601a864504c47d84b469704a")
tillage_rotary_spring_tine_glo =  Ecoinvent.get("d34f658ca1260bf83866b037f4c73306")
sowing_glo =  Ecoinvent.get("bd78d677f0e04fe1038f135719578207")

tillage_ploughing_glo = Ecoinvent.get("bcd18cd662caba362f4dcbc8e05002d7")
combine_harvesting_glo = Ecoinvent.get("20966082098d6802dc2d3e1f0346aece")


"""Biosphere outputs agri"""

# Flow to add for Carbon storage differential

#  ('Carbon dioxide, to soil or biomass stock' (kilogram, None, ('soil',)),
#  '375bc95e-6596-4aa1-9716-80ff51b9da77')

#[(flow,flow["code"]) for flow in Biosphere  if "dioxide" in flow["name"]]

Carbon_dioxide_to_soil_biomass_stock = Biosphere.get('375bc95e-6596-4aa1-9716-80ff51b9da77')






ammonia = Biosphere.get("0f440cc0-0f74-446d-99d6-8ff0e97a2444")

dinitrogen_monoxide = Biosphere.get("afd6d670-bbb0-4625-9730-04088a5b035e")

nitrogen_oxide = Biosphere.get("77357947-ccc5-438e-9996-95e65e1e1bce")

nitrogen_oxide.id

nitrate = Biosphere.get("b9291c72-4b1d-4275-8068-4c707dc3ce33")

nitrate.id



# Exchange: 3.1767e-05 kilogram 'Phosphate' (kilogram, None, ('water', 'ground-')) to 'wheat grain production' (kilogram, FR, None)>
# Phosphate
phosphate_groundwater=Biosphere.get("329fc7d8-4011-4327-84e4-34ff76f0e42d")

phosphate_groundwater.id


phosphate_surfacewater=Biosphere.get("1727b41d-377e-43cd-bc01-9eaba946eccb")

phosphate_surfacewater.id


phosphorus_surfacewater=Biosphere.get("b2631209-8374-431e-b7d5-56c96c6b6d79")

phosphorus_surfacewater.id



# Exchange: 0.0978691061631198 cubic meter 'Water' (cubic meter, None, ('air',)) to 'wheat grain production' (kilogram, FR, None)>
# Water
# 075e433b-4be4-448e-9510-9a5029c1ce94
water_air = Biosphere.get("075e433b-4be4-448e-9510-9a5029c1ce94")

water_ground = Biosphere.get("51254820-3456-4373-b7b4-056cf7b16e01")

water_surface = Biosphere.get("db4566b1-bd88-427d-92da-2d25879063b9")
water_surface.id

# Exchange: 1.42397953472488 kilogram 'Carbon dioxide, in air' (kilogram, None, ('natural resource', 'in air')) to 'wheat grain production' (kilogram, FR, None)>
# Carbon dioxide, in air
# cc6a1abb-b123-4ca6-8f16-38209df609be
carbon_dioxide_air_resource = Biosphere.get("cc6a1abb-b123-4ca6-8f16-38209df609be")


# Exchange: 15.1230001449585 megajoule 'Energy, gross calorific value, in biomass' (megajoule, None, ('natural resource', 'biotic')) to 'wheat grain production' (kilogram, FR, None)>
# Energy, gross calorific value, in biomass
# 01c12fca-ad8b-4902-8b48-2d5afe3d3a0f
energy_inbiomass = Biosphere.get("01c12fca-ad8b-4902-8b48-2d5afe3d3a0f")

# carbon dioxide fossi from urea (only wheat)
carbondioxide_fossil_urea = Biosphere.get("aa7cac3a-3625-41d4-bc54-33e2cf11ec46")

