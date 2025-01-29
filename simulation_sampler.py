# simulation_sampler.py
# Mathew Titus, April 2023
# 
# NB: Run these scripts from `pyswmm` root directory.
# 
# TODO: Create a system for formulating a large number of different
# rainfall behaviors and making them into time series for integration
# into the SWMM model training data.
# 
########################################################################

from tqdm import tqdm
import os
import itertools
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
from pyswmm import Simulation, SimulationPreConfig

import swmm_timeseries as st

# The idea here is to set up a warm-up function, priming the network to a
# 'typical' condition. These warm-up functions are constant, but could be
# linear or spiked to create different precursor conditions

# Following this, there are equal periods categorized as dry weather (0), light rain (1),
# rain (2), or heavy rain (3). Each period generates a sub-time series with random rain profile.
# The profiles are 

# To start, time series are generated at the 1-hr resolution, and have the number of days equal
# to the length of the input. Input a list of integers and the corresponding day will have a 
# total rainfall drawn from a characteristic interval ([0, 0.3] for dry weather, etc.).
# The hourly rainfall is then modeled as a sequence of iid poisson draws, or as an autocorrelated
# time series; the vector drawn is then scaled to have sum equal to the earlier r.v. sampled.

intensity_class = {
  0: {
    "min": 0.0,
    "max": 0.3,
    "start": 0.0,
    "length": 0.3
  },
  1: {
    "min": 0.3,
    "max": 0.6,
    "start": 0.3,
    "length": 0.3
  },
  2: {
    "min": 0.6,
    "max": 1.0,
    "start": 0.6,
    "length": 0.4
  },
  3: {
    "min": 1.0,
    "max": 8.0,
    "start": 1.0,
    "length": 7.0
  }
}


def get_constant(length):
  return np.ones(length);


def get_poisson(length, lam=10.0):
  '''
  lam: lambda, the intensity of the Poisson process
  '''
  return np.random.poisson(lam, length);


def get_autocorr(length):
  pass


def sample_24hr(rain_intensity, series_type="iid"):
  '''
  Returns a 24 length vector whose total rainfall (sum) lies uniformly in the
  interval defined by `intensity_class`. Vector statistics obey `series_type`
  structure.
  '''
  # hourly interval
  length = 24;

  # set total rainfall for the day (uniformly drawn from interval defined in intensity_class)
  rain_stats = intensity_class[rain_intensity]
  total_rain = rain_stats['start'] + rain_stats['length'] * np.random.rand()

  if series_type=="iid":
    series = get_poisson(length)
  elif series_type=="autocorr":
    series = get_autocorr(length)
  elif series_type=="constant":
    series = get_constant(length)

  # check that rainfall is nonnegative
  num_negs = np.where(series < 0.0)[0].shape[0]
  assert num_negs == 0, f"ERROR: Sequence with negative rainfall was generated:\n{series}"

  # scale to appropriate total rain
  L1 = np.sum(series)
  rainfall_series = (total_rain / L1) * series

  return rainfall_series;


def get_sample(raintypes):
  '''
  Generates an hourly rainfall time series with days obeying the 
  rain intensities based on the `raintypes` vector. Setting the 
  raintype to 0, 1, 2, ... for random series with the given amount of
  rain on each day. Except the first X days, which are used as a warmup.
  '''

  full_series = np.array([])
  warmup_days = 1
  try:
    assert len(raintypes) > warmup_days, "Not enough days given to `get_sample`."
  except:
    return np.array([])

  # step through each day
  for ent, typ in enumerate(raintypes):
    if ent < warmup_days:
      new_day = sample_24hr(typ, "constant")
      full_series = np.concatenate([full_series, new_day])
    else:
      new_day = sample_24hr(typ, "iid")
      full_series = np.concatenate([full_series, new_day])

  return full_series;


def generate_samples(template, rain_gage_attrs):
  '''
  Input: 
    template: Name of the input file to look for for modification,
    rain_gage_attrs={
      "names": ["RG1", "RG2", ...],
      "days": T
    }: 

  Algorithm:
    Steps through every potential assignment of rain intensity vectors, e.g. for 2 gages
    [
      {RG1: [0,0,0,0], RG2: [0,0,0,0]},
      {RG1: [0,0,0,1], RG2: [0,0,0,0]},
      {RG1: [0,0,0,2], RG2: [0,0,0,0]},
      ...,
      {RG1: [3,3,3,3], RG2: [3,3,3,3]}
    ]
    These sets are then passed into `swmm_timeseries.apply_time_series` which turns the intensity 
    assignments into time series, then the appropriate template is loaded 
    (located at "templates/`template`/{TIME SERIES length}.inp").

  Output: 
    Pickled List of SimulationPreConfig objects for creating each of the above simulation edits.
  '''
  num_gage = len(rain_gage_attrs["names"]);

  # initialize PreConfig list
  preconfigs = []

  # generate vector list
  base_elements = list(intensity_class.keys())
  premap = []
  for _ in range(rain_gage_attrs['days']):
    premap.append(base_elements)
  
  # np.repeat(base_elements, rain_gage_attrs['days'])
  sequences = list(itertools.product(*premap))
  if num_gage == 2:
    combos = list(itertools.product(sequences, sequences));
  elif num_gage == 1:
    combos = list(sequences);
  else:
    raise Exception(f"Error: rain_gage_attrs malconfigured; can only handle 1 or 2 gages as is.\n{rain_gage_attrs}");

  # generate time series
  for combo in combos:
    # initiate PreConfig object
    sim_conf = SimulationPreConfig()

    # update each raingage in turn
    if num_gage == 1:
      # get time series name
      series_name = rain_gage_attrs['names'][0]
    
      # transform couplet to a Series
      rain = pd.Series(
        reduce(
          lambda x,y: np.hstack((x,y)),
          map(sample_24hr, combo)
        )
      )

      # update `ent`th raingage
      sim_conf = st.apply_time_series(template, sim_conf, series_name, rain)

    else:
      for ent, seq in enumerate(combo):
        # get time series name
        series_name = rain_gage_attrs['names'][ent]

        # transform couplet to a Series
        rain = pd.Series(
          reduce(
            lambda x,y: np.hstack((x,y)),
            map(sample_24hr, seq)
          )
        )

        # update `ent`th raingage
        sim_conf = st.apply_time_series(template, sim_conf, series_name, rain)

    # record the new PreConfig
    preconfigs.append(sim_conf)

  return preconfigs


def step_through_samples(template_name, rain_gage_attrs, num_procs=1):
  '''
  Prep folder; generate rainfall scenarios for the template; execute each associated simulation.
  '''

  # find template file
  template = f"templates/{template_name}/{rain_gage_attrs['days']}day.inp"
  output_root = f"templates/{template_name}/{rain_gage_attrs['days']}day"

  if not os.path.exists(output_root+"/"):
    os.makedirs(output_root+"/")
    assert os.path.exists(output_root+"/"), "The `makedirs` call in `simulation_sampler` failed."
  
  print(f"Running template {template} to folder {output_root}/")

  # generate config files
  configs = generate_samples(template, rain_gage_attrs)

  # run the sample
  for ind, conf in tqdm(enumerate(configs)):
    with Simulation(template,  outputfile=f"{output_root}/{ind+64}.out", sim_preconfig = conf) as sim:
      for step in sim:
        pass


demo_rgs = {
    "names": ["TS1", "TS2"],
    "days": 3
  }


def execute_this_shit(template:str="demo_system", rgs:dict=demo_rgs, num_procs:int=1):
  '''
  Wrapper on sampler with default values.
  '''

  step_through_samples(
    template,
    rgs,
    num_procs
  )



# if __name__=="__main__":
  # input(f"Sure you want to overwrite everything in the {template} folder?")

  # run demo version
  # execute_this_shit()

  # # run WS full system sim
  # template = "ws_full"
  # rgs = { "names": ["TS1"], "days": 3 }
  # execute_this_shit(template, rgs, 8)


#




