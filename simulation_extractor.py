# import simulation_extractor as se
# simulation_extractor.py
# Mathew Titus, April 2023
# 
# Converting SWMM output to JSON data.
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
  print("running `get_data` call.")
  out = su.get_data(run_folder)

  # extract data from run
  print("running `extraction` call.")
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
  Find output at ./templates/{template_name}/{run_name}/outputs/
  '''
  print(f"ext2json call\n\ttemplate_name: {template_name}\n\trun_name: {run_name}\n\tvar_file: {var_file}")
  # find runs
  folder_name = f"templates/{template_name}"
  files = [x for x in os.listdir(f"{folder_name}/{run_name}/outputs/") if x.find(".out")>=0]

  # prep save path
  save_loc = f"templates/{template_name}/{run_name}/outputs_13step/"
  if not os.path.exists(save_loc):
    os.makedirs(save_loc)

  # iterate through run extractions
  prior_extractions = [x for x in os.listdir(save_loc) if x.find(".json")>=0]
  for run in files:
    outfile = run.replace(".out", ".json")
    # skip if already processed into this destination folder
    if outfile in prior_extractions: continue
    
    df, times = tf_prep(folder_name, f"{run_name}/outputs/{run}", var_file)
    df = df.assign(time=times)
    js = df.to_json(f"{save_loc}{outfile}", indent=1)





if __name__=="__main__":
  ext2json("ws_corrected", "3day", "var_defs.json")

  # UNUSED EXAMPLE

  # outfile = "./templates/ws_corrected/3day/outputs/2.out"
  # out = su.get_data(outfile)
  # outfile2 = "./templates/ws_corrected/3day/outputs/60.out"
  # out2 = su.get_data(outfile2)
  # with open("./templates/ws_corrected/var_defs.json", "r") as f:
  #   model_defn = json.load(f)
  
  # data = extraction(out, model_defn)
  # data2 = extraction(out2, model_defn)

  # PLOTTING

  # plottables = [x for x in data.columns if x.find("capacity")>=0]
  # rainname = [x for x in data.columns if x.find('rainfall') >= 0]
  # conduit_selection = np.random.choice(plottables, 10)

  # import matplotlib.pyplot as plt
  # f, ax = plt.subplots(3,1, figsize=(16,11))

  # data[conduit_selection].plot(kind="line", ax=ax[0])
  # ax[0].set_xlabel("Sample 1")

  # data2[conduit_selection].plot(kind="line", ax=ax[1])
  # ax[1].set_xlabel("Sample 2")

  # plt.sca(ax[2])
  # plt.plot(data['1352_rainfall'])
  # plt.plot(data2['1352_rainfall'])
  # ax[2].legend(['Sample 1', 'Sample 2'])

  # plt.savefig("figg.png")
  # plt.close()


#






