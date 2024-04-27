# swmm_timeseries.py
# Mathew Titus, April 2024
# 
# Access & edit time series for PySWMM models.
# 
######################################################################

import pyswmm
from pyswmm import Output, RainGages, SystemSeries, SimulationPreConfig


def get_times(output_path):
  '''
  Input the path to an output (.out) file.
  Fetch the datetimes indexing the simulation.
  '''
  with Output(output_path) as out:
    times = out.times

  return times


def apply_time_series(input_file, sim_conf, series_name, data): # value, row, column=2):
  '''
  Modify sim_conf to add the action:
    Replace/modify the time series `series_name` in the input file with 
    `data` DataFrame. Input must be two columns, time then value.

  NB: In sample time series, columns are
    0: Time series name
    1: Time
    2: Value
  In particular, there was no separate Date column.
  '''
  # # Create Config Handle
  # sim_conf = SimulationPreConfig()

  for _ in range(len(data)):
    datum = data.iloc[_]
    sim_conf.add_update_by_token("TIMESERIES", series_name, 1, datum.iloc[0], _)
    sim_conf.add_update_by_token("TIMESERIES", series_name, 2, datum.iloc[1], _)

  return sim_conf


