# Code documentation for the AVS_stochastic_sd repository

## Overview of the repository

This repository contains the code used to reproduce the results of the manuscript: *Jouannais.P, Marchand-Lasserre.M, Douziech.M, Pérez-López.P, When does Agrivoltaics make sense? A consequential LCA and scenario discovery approach (under review)* 

This is not a generalized platform/package to apply the procedure to any case, but the scripts specific to the procedure and not to the case (agrivoltaics) can be further adapted with additional work.

**Cite this repository:**





 
### Overview of folders and files:


**Environment**

+ **env_bw_windows.yml** File needed to create the virtual environment on WINDOWS.
+ **env_bw_ubuntu_full.yml** File needed to create the virtual environment on UBUNTU.


**PRIM_process**

Output folders where results are exported to.

**PRIM_modif**

Contains a prim script that needs to replace the original one in the prim package to allow it to work with float16.

**resultsintermediate**

Output folders where results are exported to.


**Scripts** 

+ fifteen **.py** files: python scripts including the ones to execute and the ones containing functions being called. 



Files, scripts, and their functions'interconnections are mapped below.  
<br>  

<img src="Code map.jpg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />  
<br>  


**Activities**

Collects ecoinvent activities under variable names that will be called by other scripts.

**util_functions**

Contains accessory functions.


**Stochastic_LCAs**

Computes the stochastic LCAs.


**PRIM_process**

Applies PRIM over the stochastic LCAs.

**Plot_functions**

Contains plot functions.


**Plot_from_dict**

Plots the figures with the boxes limits. 

**Parameters_and_functions**

Definition of the model's parameters and functions.

**Main_functions**

Main functions performing the LCAs and the PRIM algorithm.

**Export_double_checked_boxes**

Updates the identified boxes by only keeping and exporting the valid ones after double-checking.

**Export_double_checked_boxes**

Updates the identified boxes by only keeping and exporting the valid ones after double-checking.

**Check_boxes**

Double-check found boxes by computing the actual density of the found boxes.

**Export_boxes_xlsx**

Exports boxes into a xlsx file.

**Creation_Database_1**

Sets up the foreground database.

**Creation_Database_2**

Finishes setting up the foreground database.


**Compute_specific_configurations**

Computes and plots the LCA results of specific tested configurations.



**Setup_bw_project** 

Creates the Brightway2 project and imports ecoinvent 3.10 consequential and the biosphere.



<br>

### Reproducing results of the article

  &#128680;&#128680;&#128680;**WARNING 
The PRIM application requires substantial computing capacities (large memory/multiple CPUS). Several days of computations are necessary to reproduce the results, by using multiple instances with multiple cores. Most of the functions are written to be called in parallel with the package "ray".




*Requirements*

+ Miniconda or Anaconda
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

+ The ecoinvent license (we used ecoinvent 3.10 consquential)

+ A python interface (e.g., Spyder)


*Step by step procedure:*

1. **Download or clone a local copy of the repository. Keep the folders structure.**

2. **Prepare a conda environment with all needed packages**

+ From terminal or Anaconda/Miniconda terminal access the "Environment" folder. The path depends on where you saved the repository:

```
 cd <yourpathtothefolder/Environment>
```

+ Create the conda environment with all necessary packages using the .yml file corresponding to your OS.

**For Windows:**


```
conda env create --file env_bw_windows.yml
```

+ Activate the newly created environment:

```
conda activate env_bw_windows
```

+ Install the ray package from pip:
```
pip install ray
```

**For Ubuntu:**


```
conda env create --file env_bw_ubuntu.yml
```

+ Activate the newly created environment:

```
conda activate env_bw_ubuntu
```

+ Install the ray package from pip:
```
pip install ray
```

For MACOS, you should try to install the environment from the ubuntu file and, in case of issues, complete the environment by installing the problematic packages manually. 


3. **Replace the "prim.py" script in the ema_workbench package**
+ Copy the "prim.py" script in the PRIM_modif folder and replace the original one in the package. Find the location of the original prim file among the conda files by typing help(prim) after you import it in a python file. The modified prim file allows it to work with float16.

4. **Set up the Brigtway2 project**

+ Open the file **Setup_bw_project.py** in a text editor or python interface and change the password and username for your ecoinvent license. . 

+ From the python interface or from command, execute the whole script to prepare the Brightway2 project (```python Setup_bw_project.py```) .

+ From the python interface or from command, execute the script **Creation_Database_1.py** and then **Creation_Database_2.py** to setup the foreground database (```python Creation_Database_1.py.py```) .


5. **Compute** 

5.1 ***Compute stochastic LCAs***

+ Open the file **Stochastic_LCAs.py**. 

+ The script is parameterized for 400000 iterations. Change the number of simulations if needed and run the simulation by executing the whole script from the python interface or from command line. WIll save the intermediate results into the folder "intermediateresults".



5.2 ***Apply PRIM***

+ Open the file **PRIM_process.py**. 

+ Change the path to the corresponding files in the intermediate results generated in 4.1. If necessary, update the output file names. 

+  Apply PRIM by running the whole script. It will save the dictionaries of results into a subfolder "dict_results" in the folder "PRIM_process".


5.3 ***Check boxes and keep only valid boxes***

+ Open the file **Check_boxes.py**. 

+ Change the path to the corresponding output files with the dictionaries files generated in 5.2. If necessary, update the output file names. 

+ Execute the whole script from the python interface or from command line. This will run stochastic LCAs within the discovered boxes and export valid boxes number within "check_boxes" output files into the subfolder "Boxes" in the folder "PRIM_process".

+ Open the file **Export_double_checked_boxes.py**. 

+ Change the path to the corresponding "check_boxes" input files that were just created, and to the original dictionaries of results generated in 5.2. If necessary, update the output file names.

+ Execute the whole script from the python interface or from command line.It will save the updated dictionaries of results into a subfolder "dict_results" in the folder "PRIM_process".

5.4 ***Plot***

+ Open the file **Plot_from_dict.py**. 

+ Change the path to the input files for the result dictionaries generated in 5.3 (double-checked boxes). 

+ Execute the whole script from the python interface or from command line.It will plot the figures into a subfolder "plot_scernarios" in the folder "PRIM_process".

5.5 ***Export results under xlsx format***

+ Open the file **Export_boxes_xlsx.py**. 

+ Change the path to the input files for the result dictionaries generated in 5.3 (double-checked boxes). 

+ Execute the whole script from the python interface or from command line. It will export the results into the folder "PRIM_process".

5.6 ***Compute LCAs for specific configurations***

+ Open the file **Export_boxes_xlsx.py**. 

+ Change the path to the input files for the result dictionaries generated in 5.3 (double-checked boxes). 

+ Execute the script  **Compute_specific_configurations.py ** from the python interface or from command line. It will export the results into a csv and will plot the figures.


