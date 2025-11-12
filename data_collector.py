# data_collector.py
# Mathew Titus, STC
# November, 2025
# 
# Methods for assembling training data from SWMM outputs.
# 
################################################################

import os
import json
import datetime as dt
# specializing imports for `get_data` method
from pandas import DataFrame, concat
import numpy as np

import utils as ut


def get_data(path, run_list):
  """
  Input: Path to files, list of file numbers to load for training
  Output: Dataframes consisting of compiled file contents, separated into inputs (X) and outputs (Y)
  File number `x` corresponds to the file at [path]/[x].json
  Predictands `Y` are assumed to have "pred" in the variable name
  """
  # collect JSON data & convert to DataFrame
  data = []
  failures = []
  for run_num in run_list:
    json_path = f"{path}/{run_num}.json"
    try:
      with open(json_path, 'r') as f:
        new_json = json.load(f)
      datum = DataFrame(new_json)
      # datum = datum.assign(run_number=run_num) # better practice to keep run metadata out of the DataFrame
      data.append(datum)
    except:
      failures.append(run_num)
  
  if len(data) > 0:
    data = concat(data)
  else:
    print(f"Error: No data collected in `get_data`. Path={path}, run_list={run_list}.")
    return None
  
  if len(failures) > 0:
    print(f"Error in `get_data`: run_list included {failures} but these files failed to load.")

  # validate predictand presence before processing
  predictand_cols = list(filter(lambda x: x.find("_pred")>=0, data.columns))
  assert len(predictand_cols) > 0, f"No predictands found in column list."

  # parse times separately
  # NB: unclear what is happening to our timestamps when they're jsonified...
  # this is a rough conversion, adding 16 hours due to timezone madness
  times = data.time.apply(lambda x: dt.datetime.fromtimestamp(x/1000))\
    .reset_index(drop=True)
  df = data.drop(columns="time")\
    .reset_index(drop=True)

  # assemble predictors & predictands
  X = df.get(np.setdiff1d(df.columns, predictand_cols))
  Y = df.get(predictand_cols)

  return X, Y, times;


def get_swmm_data(template="ws_corrected", run_name="3day", data_folder="outputs_13step", num_runs=50, num_test_runs=5):
  # define paths - to be run from repo root
  (path2runs,) = ut.populate_paths(template, run_name, [data_folder])

  # collect all output files
  run_list = os.listdir(path2runs)
  run_list = [int(x.rstrip(".json")) for x in run_list if (x[-5:]==".json")]

  assert len(run_list) >= num_runs, f"There are {len(run_list)} runs available, but {num_runs} training runs requested."
  assert ((len(run_list) - num_runs) >= num_test_runs), f"Only {len(run_list)-num_runs} non-training runs available. \
    {num_test_runs} test runs requested. {num_runs} training runs. Total of {len(run_list)} runs."

  # split training & test data
  runs4training = np.random.choice(run_list, num_runs)
  run4testing = np.random.choice(np.setdiff1d(run_list, runs4training), num_test_runs)

  # fetch & assemble
  training_data = get_data(path2runs, runs4training)
  test_data = get_data(path2runs, run4testing)

  return (runs4training, training_data, run4testing, test_data);


#








