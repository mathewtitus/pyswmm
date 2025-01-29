# convert2meso.py
# Mathew Titus, Sunstrand Technical Consulting
# December, 2024
# 
# Uses high-resolution network simulation results to create a
# "projected" version of the simulation with fewer variables.
# 
################################################################################


import pandas as pd
import numpy as np
import json
import os


run = "3day" # define sim length of interest
var_type = ["_capacity"]
suffixes = ["_m0", "_m1", "_m2", "_m3", "_m4", "_pred"]

# get SWMM output data: meso version uses same sim output as ws_full
sim_folder = f'/home/titusm/GitHub/pyswmm/templates/ws_full/{run}/outputs/'
sim_files = [x for x in os.listdir(sim_folder) if os.path.splitext(x)[1]==".json"]

# load example dataset
with open(sim_folder+"24.json", "r") as f:
  data = json.load(f)

data = pd.DataFrame.from_dict(data)

# get network data
nw_folder = '/home/titusm/GitHub/pyswmm/templates/ws_simple/'
nw_files = [x for x in os.listdir(nw_folder) if os.path.splitext(x)[1]==".csv"]

meso_data = pd.DataFrame(index=data.index)
for (_ind, _file) in enumerate(nw_files):
  df = pd.read_csv(nw_folder+_file, index_col=0)

  # collect columns of interest
  var_list = ['time']; # list(zip(df.Name, var_type, suffixes))
  rain_vars = [x for x in data.columns if x.find("rainfall") >= 0]
  var_list.extend(rain_vars)

  # TODO: Improve looping
  for nom in df.Name.tolist():
    new_vars = [nom+var_type[0]+suff for suff in suffixes]
    var_list.extend(new_vars)

  micro_data = data.get(var_list)
  
  # for each mesonetwork element, aggregate data according to conduit dims


  # add aggregated data points to meso datafile
  meso_data.insert(_ind, _file.split('.')[0], meso_var)


######### NB: #########
# (Loading a sample SWMM output (3day, 24.json) we get 4668 columns, 6 of which correspond to rain/time)
# (Each remaining var should have 6 entries: m0 - m4, & pred. Counting, we have (4668-6)/6 = 777 distinct elements)
# (But counting the elements in ICM, we see 338 nodes + 386 links = 724 elements. What are the additional 53 vars???)
######### /NB #########




