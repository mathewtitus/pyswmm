# import simulation_extractor as se
# simulation_extractor.py
# Mathew Titus, April 2023
# 
# TODO: Add a check in ext2json so if a file has already 
# been converted to output it gets skipped.
# 
########################################################################

import os
import json
import numpy as np
import datetime as dt
from functools import reduce
# import matplotlib.pyplot as plt
import pandas as pd
from pyswmm import Simulation, SimulationPreConfig

import swmm_timeseries as st
import swmm_utils as su


machine = "stoner"; # set to "stoner" when on PC; set to "mac" o/w

def get_local_dir(machine="mac"):
  if machine == "mac":
    return "/Users/mtitus/Documents/GitHub/COS_WW/pyswmm/";
  elif machine == "stoner":
    return "/home/titusm/GitHub/pyswmm/";


def extraction(out, model_defn):
  '''
  Collect variable I/O to train with from helper JSON file in [template_name] folder
  Helper lists `input` and `output` variables by category. 
  `time_steps` gives number of steps to include in input from present back
  `horizon` gives how many time steps ahead the model will predict
  '''

  # compile dataset columns
  df = pd.DataFrame([], index=pd.Index([], name="time"))
  for cat in model_defn['input'].keys():
    variates = model_defn['input'][cat]
    elements = list(out[cat])
    for elem in elements:
      sub_df = out[cat][elem].get(variates)
      colm_map = dict(map(
          lambda v: [v, "_".join([elem, v])],
          np.setdiff1d(variates, ["time"])
      ))
      sub_df = sub_df.rename(columns=colm_map)\
        .set_index('time')
      df = df.combine_first(sub_df)

  ## flag warmup days
  # get sim start time
  t0 = np.min(df.index)
  # get trial start time
  t1 = t0 + dt.timedelta(days=model_defn['warmup_days'])
  # create flag
  df = df.assign(warmup=(df.index < t1))

  # NB: May want to change in the future: dropping all but one rainfall time series
  rain_cols = filter(lambda c: c.find("rainfall")>=0, df.columns)
  # drop first column name
  rain_cols.__next__()
  # drop rest of columns
  df = df.drop(columns=list(rain_cols))

  return df;


def tf_prep(template_name, swmm_outfile, var_file):
  '''
  Input:
    subtemplate_name: templates/[template], AKA: the run (e.g. 3day, 5day, ...)
    swmm_outfile: the outfile name (e.g. 0.out, 1.out, ...)
    var_file: where the different network elements' variables of interest are defined. See templates/demo_system/var_defs.json for an example.
  Output:
    ...
  '''
  print(f"tf_prep call\n\ttemplate_name: {template_name}\n\tswmm_outfile: {swmm_outfile}\n\tvar_file: {var_file}")
  tem_folder = f"{get_local_dir(machine)}{template_name}/"
  run_folder = f"{tem_folder}{swmm_outfile}"
  print(f"\ttem_folder: {tem_folder}")
  print(f"\trun_folder: {run_folder}")

  # get variable definitions from var_file
  with open(f"{tem_folder}{var_file}", "r") as f:
    model_defn = json.load(f)

  # get output from model
  out = su.get_data(run_folder)

  # extract data from run
  df = extraction(out, model_defn)
  
  # find starting position
  ind0 = np.max(np.where(df.warmup)[0])
  # save times for later
  times = df.index.to_numpy()[ind0+1:]
  # steps to horizon
  horizon = model_defn['horizon']
  # steps as predictors
  presteps = model_defn['time_steps']
  # drop warmup column
  df = df.drop(columns="warmup")

  tf_data = []
  for ind in np.arange(ind0, df.shape[0]-horizon):
    predictors = df.iloc[ind-presteps+1:ind+1]
    # flatten predictors dataframe
    cols = df.columns.to_numpy();
    var_index = reduce(
      lambda x, y: np.hstack((x, y)),
      map(
        lambda m: cols + "_m"+str(m),
        np.arange(presteps-1,-1,-1) # countdown
      ))
    predictors = pd.DataFrame(data=predictors.to_numpy().flatten().reshape((1,-1)), columns=var_index)

    # don't try to predict rainfall
    predictand = df.drop(columns=list(filter(lambda x: x.find("rainfall")>=0, df.columns)))
    # select time step to predict (converts to Series object)
    predictand = predictand.iloc[ind+horizon]
    # add "pred" column suffix
    predictand = predictand.rename(dict(map(lambda x: [x, x+"_pred"], predictand.index)))
    # revert to dataframe
    predictand = predictand.to_frame().T.reset_index(drop=True)

    # return predictors, predictand

    tf_data.append(predictors.combine_first(predictand))
    # print(f"index {ind}\n{predictors.combine_first(predictand)}")

  # return tf_data
  tf_data = pd.concat(tf_data)

  return tf_data.reset_index(drop=True), times


def ext2json(template_name, run_name, var_file):
  '''
  Goes to run folder, selects each output in turn and converts to JSON
  Find output at ./templates/{template_name}/outputs/{run_name}/
  '''
  print(f"ext2json call\n\ttemplate_name: {template_name}\n\trun_name: {run_name}\n\tvar_file: {var_file}")
  # find runs
  folder_name = f"templates/{template_name}"
  files = [x for x in os.listdir(f"{folder_name}/{run_name}") if x.find(".out")>=0]

  # prep save path
  save_loc = f"templates/{template_name}/outputs/{run_name}/"
  if not os.path.exists(save_loc):
    os.makedirs(save_loc)

  # iterate through run extractions
  for run in files:
    outfile = run.replace(".out", ".json")
    df, times = tf_prep(folder_name, f"{run_name}/{run}", var_file)
    df = df.assign(time=times)
    js = df.to_json(f"{save_loc}{outfile}", indent=1)





if __name__=="__main__":
  ext2json("ws_full", "3day", "var_defs.json")



#






