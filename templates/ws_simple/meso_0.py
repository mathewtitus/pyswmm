# meso_0.py
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
monitor_nodes = ["36-484057", "36-486016", "33-488083", "36-486002", "36-486023"]

# get SWMM output data: meso version uses same sim output as ws_full
sim_folder = f'/home/titusm/GitHub/pyswmm/templates/ws_full/{run}/outputs/'
sim_files = [x for x in os.listdir(sim_folder) if os.path.splitext(x)[1]==".json"]

for _sf in sim_files:
  # load example dataset
  with open(sim_folder+_sf, "r") as f:
    data = json.load(f)

  data = pd.DataFrame.from_dict(data)

  # drop nodes that aren't associated to a monitor:
  node_cols = [x for x in data.columns if x.find("invert_depth")>=0]
  drop_cols = list(filter(
    lambda x: ~np.any([(y+"_invert_depth" in x) for y in monitor_nodes]),
    node_cols
  ))

  meso_data = data.drop(columns=drop_cols)

  with open(f'/home/titusm/GitHub/pyswmm/templates/ws_simple/{run}/outputs/{_sf}', 'w+') as f:
    json.dump(meso_data.to_dict(), f, indent=1)


