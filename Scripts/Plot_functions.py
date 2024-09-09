# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:32:56 2024

@author: pierre.jouannais
"""

import matplotlib as mplt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import datetime
from time import *
import re


import util_functions






def count_overlapping_intervals(normalized_intervals,
                                value):
    
    """Counts the number of overlapping intevals at a certain value (normalized)"""
    
    count = 0
    for interval in normalized_intervals:
        if interval[0] <= value <= interval[1]:
            count += 1
    return count









def clean_last_box(list_boxes_limits_df,
                   min_density):
    
    """The algorithm returns one additional box above the threshold. The function cleans it away."""

    list_boxes_limits_df_copy = list_boxes_limits_df[1:]


    list_boxes_limits_df_copy = [box for box in list_boxes_limits_df_copy if box["density"]>=min_density]
    
    return list_boxes_limits_df_copy

def collect_cumul_cover(list_boxes_limits_df):
    
    """Collects the cumulated cover, in the last box """
    
    print(list_boxes_limits_df)
    cumul_cover = 0
    
    if len(list_boxes_limits_df)!=0:
        cumul_cover=list_boxes_limits_df[-1]["cover_cumul"]
        
    return cumul_cover




def marker_qp(value, 
              position):
    
    """ Returns the marker depending on the qp_value and the position of the border"""
    
    if position == "min":
        if value == 1:
            return '<' 
        elif value == 0:
            return '|'  
        else:
            return "." 
    elif position == "max":
        if value == 1:
            return '>'  
        elif value == 0:
            return '|'  
        else:
            return "."  




def marker_qp_switch(value):
    
    """ Returns the marker depending on the qp_value and the position of the border"""
    
    if value !=0:
        return '.' 
    else:
        return '|'  




def size_marker_qp(value):
    
    """ Returns the size of the marker depending on the qp_value"""
    
    if value == 1: # size of the triangle is fixed
        size = 5
    
    elif value == 0:  # size of the tick is fixed
        size = 5
    
    else: # size of the diamond varies according to qp
        size = 5 * (10 - value * 10)
        
    return size    

        
    
        
        
        
# def plot_intervals_non_normalized_fix_switch(min_density,
#                                              sample_dataframe, 
#                                              list_boxes_limits_df,
#                                              key,
#                                              cutoff, 
#                                              res,
#                                              colnames_switch):
    
#     """ Plots the intervals for each scenario, each parameter"""
    
#     sample_dataframe_only_variables = sample_dataframe.copy()
#     sample_dataframe_only_variables.drop([col for col in sample_dataframe_only_variables.columns if len(pd.unique(sample_dataframe_only_variables[col])) == 1], axis=1, inplace=True)
    
#     #Potentially remove the last box if ddoes not meet density criteria
#     list_boxes_limits_df=clean_last_box(list_boxes_limits_df,min_density)
    
#     cover_cumul = collect_cumul_cover(list_boxes_limits_df)
    
#     num_scenarios = len(list_boxes_limits_df)

#     avg_interval_sizes = {}
#     for param_name in list(sample_dataframe_only_variables.columns):
        
#         if "switch" not in param_name:   ################################################################################ TO BE CHANGED TO SWITCH IN NAME
                
#             interval_sizes = []
            
#             for scenario_data in list_boxes_limits_df:
#                 if param_name in scenario_data:
#                     interval = scenario_data[param_name]
#                     normalized_interval_min = (interval[0] - min(sample_dataframe_only_variables[param_name])) / (max(sample_dataframe_only_variables[param_name]) - min(sample_dataframe_only_variables[param_name]))
#                     normalized_interval_max = (interval[1] - min(sample_dataframe_only_variables[param_name])) / (max(sample_dataframe_only_variables[param_name]) - min(sample_dataframe_only_variables[param_name]))
                    
#                     interval_sizes.append(normalized_interval_max - normalized_interval_min)
#             avg_interval_sizes[param_name] = np.mean(interval_sizes) if interval_sizes else 0
#         else:
#             avg_interval_sizes[param_name] = 1  # This puts the switch parameters at the end of the plot
            
#     print(avg_interval_sizes)
    
#     # Sort parameters based on average interval size
#     sorted_params = sorted(avg_interval_sizes, key=lambda x: avg_interval_sizes[x])
#     print(sorted_params)
    
#     sorted_params_cutoff = [param for param in sorted_params if avg_interval_sizes[param] <= cutoff]
    
#     cutoffupdate_total = cutoff
    
#     while len(sorted_params_cutoff) < 2:
#         cutoffupdate_total = cutoffupdate_total + 0.1
#         sorted_params_cutoff = [param for param in sorted_params if avg_interval_sizes[param] <= cutoffupdate_total]

#     print("cutoffupdate_total", cutoffupdate_total)    
        
#     print(sorted_params_cutoff)
    
#     print("above cutoff", len(sorted_params_cutoff))

#     # Create a color map for scenarios
#     color_map = plt.get_cmap('tab20')
#     scenario_colors = [color_map(i) for i in np.linspace(0, 1, num_scenarios)]

#     # Create a figure with subplots arranged in a grid of 5 columns
#     num_rows = (len(sorted_params_cutoff) + 4) // 5
    
#     if len(sorted_params_cutoff) < 5:
#         num_cols = len(sorted_params_cutoff)
#     else:
#         num_cols = 5
                
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows), sharex=False, dpi=res)
    

    
#     #Add title to the entire figure
#     fig.suptitle('Cumulated cover '+str(cover_cumul), fontsize=16)
#     # Flatten the axs array for easy iteration
#     axs = axs.flatten()
#     print("heeere")
    
#     # Plot "overlap count" and "intervals" for each parameter
#     for idx, param_name in enumerate(sorted_params_cutoff):
#         param_range = [min(sample_dataframe_only_variables[param_name]), max(sample_dataframe_only_variables[param_name])]
#         print("uuuu")

#         # Calculate normalized intervals for the parameter
#         intervals = []
#         qp_values = []  # Store qp values for dot size
#         for scenario_data in list_boxes_limits_df:
#             if param_name in scenario_data:
#                 interval = scenario_data[param_name]

#                 intervals.append([interval[0], interval[1]])

#         # Create linspace vector between 0 and 1
#         x_vals = np.linspace(param_range[0], param_range[1], 1000)
        
#         # Plot "overlap count" curve
#         ax_overlap = axs[idx]
#         if "switch" not in param_name:

#             overlap_values = [count_overlapping_intervals(intervals, val) for val in x_vals]
#             ax_overlap.plot(x_vals, overlap_values, color='green')

#             ax_overlap.set_yticks(np.arange(min(overlap_values), max(overlap_values) + 1, 1))
#             ax_overlap.set_ylabel('')

#         ax_overlap.set_title(f'{param_name} ')
#         # Plot intervals for each scenario
#         ax_intervals = ax_overlap.twinx()
#         ax_intervals.set_ylabel('Box')

#         for i, scenario_data in enumerate(list_boxes_limits_df):
#             if param_name in scenario_data:
#                 interval = scenario_data[param_name]
#                 interval_min = interval[0] 
#                 interval_max = interval[1] 
#                 qp_value_min = interval[2]
#                 qp_value_max = interval[3]


        
        
        
        
#                 ax_intervals.plot([interval_min, interval_max], [i + 1, i + 1], color=scenario_colors[i],linewidth=scenario_data["cover"]*20)
#                 # Add dots with sizes based on qp-values
#                 ax_intervals.scatter(interval_min, i + 1, 
#                                      s=size_marker_qp(qp_value_min),
#                                      marker=marker_qp(qp_value_min,"min"), 
#                                      color=scenario_colors[i], alpha=1)
                
#                 ax_intervals.scatter(interval_max, i + 1, 
#                                      s=size_marker_qp(qp_value_max),
#                                      marker=marker_qp(qp_value_max,"max"), 
#                                      color=scenario_colors[i], alpha=1)                
#         # Remove grid lines and set background color
#         ax_overlap.grid(False)
#         ax_intervals.grid(False)
#         ax_overlap.set_facecolor('white')
#         ax_intervals.set_facecolor('white')
#         ax_intervals.set_yticks(np.arange(1, num_scenarios + 1))
        
        
#         intervals_ticks = np.round(np.linspace(param_range[0], param_range[1], 5), 2)
        
        
#         #print(intervals_ticks)
        
#         if "switch" not in param_name:
#             #print("switch",param_name)
#             ax_overlap.set_xticks(intervals_ticks)
#             ax_overlap.set_xticklabels([str(a) for a in intervals_ticks])
            
#         else:
#             print("switch",param_name)
#             ax_overlap.set_yticks([])
#             ax_overlap.set_yticklabels([])
#             ax_overlap.set_xticks(list(range(len(colnames_switch))))
#             ax_overlap.set_xticklabels([str(a) for a in colnames_switch])
            
            
#     # Hide any extra subplots if the number of parameters is not a multiple of 5
#     for ax in axs[len(sorted_params_cutoff):]:
#         ax.axis('off')

#     fig.tight_layout() # used to be plt
#     #Adjust the top margin to move the title above the figure
#     fig.subplots_adjust(top=0.95)
#     # Save the figure in the root folder
#     x = datetime.datetime.now()

#     month = str(x.month)
#     day = str(x.day)
#     microsec = str(x.strftime("%f"))
    
#     name_fig = "sc_" + microsec 
    
#     name_folder = '../PRIM_process/plot_scenarios/'+key+month+microsec
    
#     util_functions.create_folder_if_not_exists(name_folder)
    
#     plt.savefig(name_folder+"/" + name_fig + '.png')
#     plt.clf()
#     plt.cla()
#     plt.close()             
    
#     # plt.show()



########################################

#########################




def extract_density_value(input_string):
    
    """Extracts the density value used by PRIM for the analyzed sample"""

    # Define the pattern to match "density_" followed by a numerical value before "peel"
    pattern = r'density_(\d+\.\d+)peel'

    # Use regular expression to search for the pattern in the input string
    match = re.search(pattern, input_string)

    # Check if a match is found
    if match:
        # Extract the matched value and convert it to a float
        density_value = float(match.group(1))
        return density_value
    else:
        return None






# def plot_from_dict_results_fix_switch(dict_results_all_main,
#                                       sample_dataframe,
#                                       res,
#                                       cutoff,
#                                       list_meth,
#                                       colnames_switch,
#                                       typesim):


#     """Calls the plotting function over the different keys of the result dictionnaries"""
    
#     dict_results_all_maincopy=dict_results_all_main.copy()
#     info=dict_results_all_maincopy.pop("info")
#     main=info["main"]
    
#     for key in dict_results_all_maincopy:

#         print(key)
#         min_density=extract_density_value(key)
#         print(min_density)
        
#         subselection_dict = dict_results_all_maincopy[key]
        
#         [
          
#           list_boxes_limits_df_all_meth,
          
#           list_boxes_limits_df_total,
#           recombine_table_all] =subselection_dict
        
#         print("xxx")
#         name_plot = key+main+typesim+"total_nocutoff"
#         plot_intervals_non_normalized_fix_switch(min_density,
#                                         sample_dataframe,
#                            list_boxes_limits_df_total,
#                            name_plot,
#                            1,
#                            res,
#                            colnames_switch)
#         print("Plotted", name_plot)

#         print("xxx")
#         name_plot = key+typesim+"total_cutoff"+str(cutoff)+main
#         plot_intervals_non_normalized_fix_switch(min_density,
#                                         sample_dataframe,
#                            list_boxes_limits_df_total,
#                            name_plot,
#                            cutoff,
#                            res,
#                            colnames_switch)
#         print("Plotted", name_plot) 
        
        
#         for meth_index in range(len(list_meth)):
            
#             name_meth= list_meth[meth_index]
#             meth_code = name_meth[-1]
#             list_boxes_limits_df = list_boxes_limits_df_all_meth[meth_index]
           
#             name_plot = key+main+typesim+meth_code+"_nocutoff"
            
#             plot_intervals_non_normalized_fix_switch(min_density,
#                                             sample_dataframe,
#                                list_boxes_limits_df,
#                                name_plot,
#                                1,
#                                res,
#                                colnames_switch)
            
#             print("Plotted", name_plot)
            
#             name_plot = key+main+typesim+meth_code+"cutoff"+str(cutoff)

#             plot_intervals_non_normalized_fix_switch(min_density,
#                                             sample_dataframe,
#                                list_boxes_limits_df,
#                                name_plot,
#                                cutoff,
#                                res,
#                                colnames_switch)            
            
            
            
#             print("Plotted", name_plot)






##### with categ


def plot_intervals_non_normalized_fix_switch_categ(min_density,
                                             sample_dataframe, 
                                             list_boxes_limits_df,
                                             key,
                                             cutoff, 
                                             res,
                                             colnames_switch):
    
    """ Plots the intervals for each scenario, each parameter"""
    
    sample_dataframe_only_variables = sample_dataframe.copy()
    sample_dataframe_only_variables.drop([col for col in sample_dataframe_only_variables.columns if len(pd.unique(sample_dataframe_only_variables[col])) == 1], axis=1, inplace=True)
    
    #Potentially remove the last box if ddoes not meet density criteria
    list_boxes_limits_df=clean_last_box(list_boxes_limits_df,min_density)
    
    cover_cumul = collect_cumul_cover(list_boxes_limits_df)
    
    num_scenarios = len(list_boxes_limits_df)

    avg_interval_sizes = {}
    for param_name in list(sample_dataframe_only_variables.columns):
        
        if "switch" not in param_name:   ################################################################################ TO BE CHANGED TO SWITCH IN NAME
                
            interval_sizes = []
            
            for scenario_data in list_boxes_limits_df:
                if param_name in scenario_data:
                    interval = scenario_data[param_name]
                    normalized_interval_min = (interval[0] - min(sample_dataframe_only_variables[param_name])) / (max(sample_dataframe_only_variables[param_name]) - min(sample_dataframe_only_variables[param_name]))
                    normalized_interval_max = (interval[1] - min(sample_dataframe_only_variables[param_name])) / (max(sample_dataframe_only_variables[param_name]) - min(sample_dataframe_only_variables[param_name]))
                    
                    interval_sizes.append(normalized_interval_max - normalized_interval_min)
            avg_interval_sizes[param_name] = np.mean(interval_sizes) if interval_sizes else 0
        else:
            avg_interval_sizes[param_name] = 1  # This puts the switch parameters at the end of the plot
            
    print("avg intervale sizes",avg_interval_sizes)
    
    # Sort parameters based on average interval size
    sorted_params = sorted(avg_interval_sizes, key=lambda x: avg_interval_sizes[x])
    
    print("sorted params",sorted_params)
    
    sorted_params_cutoff = [param for param in sorted_params if avg_interval_sizes[param] <= cutoff]
    
    cutoffupdate_total = cutoff
    
    while len(sorted_params_cutoff) < 2:
        cutoffupdate_total = cutoffupdate_total + 0.1
        sorted_params_cutoff = [param for param in sorted_params if avg_interval_sizes[param] <= cutoffupdate_total]

    print("cutoffupdate_total", cutoffupdate_total)    
        
    print(sorted_params_cutoff)
    
    print("above cutoff", len(sorted_params_cutoff))

    # Create a color map for scenarios
    color_map = plt.get_cmap('tab20')
    scenario_colors = [color_map(i) for i in np.linspace(0, 1, num_scenarios)]

    # Create a figure with subplots arranged in a grid of 5 columns
    num_rows = (len(sorted_params_cutoff) + 4) // 5
    
    if len(sorted_params_cutoff) < 5:
        num_cols = len(sorted_params_cutoff)
    else:
        num_cols = 5
                
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows), sharex=False, dpi=res)
    

    
    #Add title to the entire figure
    fig.suptitle('Cumulated cover '+str(cover_cumul), fontsize=16)
    # Flatten the axs array for easy iteration
    axs = axs.flatten()
    print("heeere")
    
    # Plot "overlap count" and "intervals" for each parameter
    for idx, param_name in enumerate(sorted_params_cutoff):
        
        if len(list_boxes_limits_df)>100:
            interval_ticks = 15
        else:
            interval_ticks = 5
        
        
        if "switch" not in param_name:
            
            param_range = [min(sample_dataframe_only_variables[param_name]), max(sample_dataframe_only_variables[param_name])]
            print("uuuu")
        else:
            
            param_range=["ALL"]
            
        # Calculate normalized intervals for the parameter
        intervals = []
        qp_values = []  # Store qp values for dot size
        for scenario_data in list_boxes_limits_df:
            if param_name in scenario_data:
                interval = scenario_data[param_name]

                intervals.append([interval[0], interval[1]])



        # Plot "overlap count" curve
        ax_overlap = axs[idx]
        if "switch" not in param_name:
            
            # Create linspace vector
            x_vals = np.linspace(param_range[0], param_range[1], 1000)
            
            overlap_values = [count_overlapping_intervals(intervals, val) for val in x_vals]
            ax_overlap.plot(x_vals, overlap_values, color='green')

            ax_overlap.set_yticks(np.arange(min(overlap_values), max(overlap_values) + 1, 1))
            
            
            y_overlaplabels = [str(i + 1) if (i + 1) % interval_ticks == 0 else '' for i in np.arange(min(overlap_values), max(overlap_values) + 1, 1)]
            ax_overlap.set_yticklabels(y_overlaplabels)        
            
            #ax_overlap.set_ylabel('Count of Overlapping Intervals')
            ax_overlap.set_ylabel('')

        ax_overlap.set_title(f'{param_name} ')
        # Plot intervals for each scenario
        ax_intervals = ax_overlap.twinx()
        ax_intervals.set_ylabel('Box')

        for i, scenario_data in enumerate(list_boxes_limits_df):
            if param_name in scenario_data:
                interval = scenario_data[param_name]
                interval_min = interval[0] # for switch/categorical parameters, the names of all the categories
                interval_max = interval[1] # for switch/categorical parameters, the names of all the categories (repeated)
                qp_value_min = interval[2]
                qp_value_max = interval[3]


        
        
        
                if "switch" not in param_name:
                    
                    ax_intervals.plot([interval_min, interval_max], [i + 1, i + 1], color=scenario_colors[i], linewidth=0.8)
                    # Add dots with sizes based on qp-values
                    ax_intervals.scatter(interval_min, i + 1, 
                                         s=size_marker_qp(qp_value_min),
                                         marker=marker_qp(qp_value_min,"min"), 
                                         color=scenario_colors[i], alpha=1)
                    
                    ax_intervals.scatter(interval_max, i + 1, 
                                         s=size_marker_qp(qp_value_max),
                                         marker=marker_qp(qp_value_max,"max"), 
                                         color=scenario_colors[i], alpha=1)    
                    
                else:
                    # the names of all the selected categories are in interval min (and also in interval max)
                    
                    for cat in interval_min:
                    
                        # Add dots with sizes based on qp-values
                        ax_intervals.scatter(cat, i + 1, 
                                             s=size_marker_qp(qp_value_min), 
                                             marker= marker_qp_switch(qp_value_min),
                                             color=scenario_colors[i], alpha=1)
                        
                       
                    
                    
        # Remove grid lines and set background color
        ax_overlap.grid(False)
        ax_intervals.grid(False)
        ax_overlap.set_facecolor('white')
        ax_intervals.set_facecolor('white')
        #ax_intervals.set_yticks(np.arange(1, num_scenarios + 1))
        ax_intervals.set_yticks(np.arange(1, num_scenarios + 1))
        

        y_labels = [str(i + 1) if (i + 1) % interval_ticks == 0 else '' for i in range(num_scenarios)]
        ax_intervals.set_yticklabels(y_labels)        
        
        
        
        #print(intervals_ticks)
        
        if "switch" not in param_name:
            #print("switch",param_name)
            if count_digits_in_float(param_range[0])>=4:

                intervals_ticks = np.round(np.linspace(param_range[0], param_range[1], 5), 2)
            else:
                intervals_ticks = np.round(np.linspace(param_range[0], param_range[1], 7), 2)

                

            ax_overlap.set_xticks(intervals_ticks)
            ax_overlap.set_xticklabels([str(a) for a in intervals_ticks])
            
        else:
            print("switch",param_name)
            ax_overlap.set_yticks([])
            ax_overlap.set_yticklabels([])
            # ax_overlap.set_xticks(list(range(len(colnames_switch))))
            # ax_overlap.set_xticklabels([str(a) for a in colnames_switch
                                       #  ]
                                       
                                       # )
            
            
    # Hide any extra subplots if the number of parameters is not a multiple of 5
    for ax in axs[len(sorted_params_cutoff):]:
        ax.axis('off')

    fig.tight_layout() # used to be plt
    #Adjust the top margin to move the title above the figure
    fig.subplots_adjust(top=0.95)
    # Save the figure in the root folder
    x = datetime.datetime.now()

    month = str(x.month)
    day = str(x.day)
    microsec = str(x.strftime("%f"))
    
    name_fig = "sc_" + microsec 
    
    name_folder = '../PRIM_process/plot_scenarios/'+key+month+microsec
    
    util_functions.create_folder_if_not_exists(name_folder)
    
    plt.savefig(name_folder+"/" + name_fig + '.png')
    plt.clf()
    plt.cla()
    plt.close()             
    
    # plt.show()









def plot_from_dict_results_fix_switch_categ(dict_results_all_main,
                                      sample_dataframe,
                                      res,
                                      cutoff,
                                      list_meth,
                                      colnames_switch,
                                      typesim):


    """Calls the plotting function over the different keys of the result dictionnaries"""
    
    dict_results_all_maincopy=dict_results_all_main.copy()
    info=dict_results_all_maincopy.pop("info")
    main=info["main"]
    
    for key in dict_results_all_maincopy:

        print(key)
        min_density=extract_density_value(key)
        print(min_density)
        
        subselection_dict = dict_results_all_maincopy[key]
        
        [
          
          list_boxes_limits_df_all_meth,
          
          list_boxes_limits_df_total,
          recombine_table_all] =subselection_dict
        
        print("xxx")
        name_plot = key+main+typesim+"total_nocutoff"
        plot_intervals_non_normalized_fix_switch_categ(min_density,
                                        sample_dataframe,
                           list_boxes_limits_df_total,
                           name_plot,
                           cutoff,
                           res,
                           colnames_switch)
        print("Plotted", name_plot)


        
        for meth_index in range(len(list_meth)):
            
            name_meth= list_meth[meth_index]
            meth_code = name_meth[-1]
            list_boxes_limits_df = list_boxes_limits_df_all_meth[meth_index]
           
            name_plot = key+main+typesim+meth_code+"_nocutoff"
            
            plot_intervals_non_normalized_fix_switch_categ(min_density,
                                            sample_dataframe,
                               list_boxes_limits_df,
                               name_plot,
                               cutoff,
                               res,
                               colnames_switch)
            
            print("Plotted", name_plot)
            



def count_digits_in_float(x):
    # Convert the float to a string
    float_str = str(x)
    
    # Remove the decimal point
    float_str = float_str.replace('.', '')
    
    # Count the number of digits
    num_digits = len(float_str)
    
    return num_digits



