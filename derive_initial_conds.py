# derive_initial_conds.py
# Mathew Titus, May 2025
# Sunstrand Technical Consulting
# 
# 
# 
########################################################

import os
import json
import itertools
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

from pyswmm import Simulation, Nodes, Links, \
  SubcatchSeries, NodeSeries, LinkSeries, \
  RainGages, SystemSeries, SimulationPreConfig, Subcatchments

import swmm_utils as su
import swmm_timeseries as st
import simulation_sampler as ss

on_vm = os.path.abspath('.').find("/home/ubuntu/") >= 0

# get rain history
if on_vm:
  rain_path = "/home/ubuntu/rain/latest.json"
else:
  rain_path = "/Users/mtitus/Documents/GitHub/COS_WW/rain/latest.json"

# get prism data
if on_vm:
  prism_path = "/home/ubuntu/cos-ww-remote/data/prism/"
else:
  prism_path = "/Users/mtitus/Documents/GitHub/COS_WW/cos-ww-remote/data/prism/"

num = 19
filename = f"WS_{num}_latest.json"
with open(prism_path+filename, 'r') as f:
  prism = json.load(f)

# start with dry weather state



