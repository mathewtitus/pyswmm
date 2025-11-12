# utils.py
# Mathew Titus, November 2025
# Sunstrand Technical Consulting
# 
# Support Functions for NEWC ML
# 
##########################################

import os
import platform
import datetime as dt


class WorkingDirException(Exception):
  '''
  Exception class for when the correct `pyswmm`
  repository root cannot be determined.
  '''
  def __init__(self, message):
    super().__init__(message)
    self.error = "You are executing a call from a machine that cannot be determined by \
    `pyswmm.utils`. You may need to update `pyswmm.utils.get_root` if system paths are \
    out of date."

  def __str__(self):
    return self.error


def get_root():
  '''
  Determines system path to repository root.
  Defined for Mat's macs, pc, or cosww-remote vm.
  Returned path ends in '/' character.
  '''
  here = os.path.abspath(".")

  # determine machine; define root
  if here.find("ubuntu") > 0:
    # on VM
    root = "/home/ubuntu/"
  elif os.sys.platform == "darwin":
    if platform.processor() == "i386":
      # on old mac
      root = "/Users/mtitus/Documents/GitHub/COS_WW/pyswmm/"
    elif platform.processor() == "arm":
      # on new mac
      root = "/Users/mtitus/Documents/GitHub/COSWW/pyswmm/"
    else:
      raise WorkingDirException()
  elif platform.uname().node == "Stoner":
    # on stoner
    root = "/home/titusm/GitHub/pyswmm/"
  else:
    raise WorkingDirException()

  return root


def populate_paths(template, run_name, folders):
  '''
  returns list of paths based on scenario (defined by `template` and `run_name`)
  and the paths requested (listed in `folders`).
  NB: Execution will create a subfolder for each path requested by the `folders` arg.

  Typical usage:
    path2runs = f"templates/{template}/{run_name}/outputs_13step"
    path2models = f"templates/{template}/{run_name}/models"
    path2figs = f"templates/{template}/{run_name}/figures"
    path2perf = f"templates/{template}/{run_name}/performance"
  '''
  assert isinstance(folders, list), f"Argument `folders` must be a list, but was of type {type(folders)}";

  # define common path
  subroot = f"templates/{template}/{run_name}/"

  # setup file paths
  paths = [subroot + _f for _f in folders]
  for _path in paths:
    if not os.path.exists(_path):
      os.makedirs(_path)
  
  return paths


def now_timestamp():
  return str(dt.datetime.now().timestamp()).split('.')[0]


def now_string():
  return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


#


import numpy as np

def get_loss(Y2, Y2p):
  # calculate residuals
  err = Y2p - Y2 # signed numpy array
  L2 = np.sqrt(np.sum(err**2, axis=1))
  L2_spatial = np.sqrt(np.sum(err**2, axis=0))
  Linfty_spatial = np.max(np.abs(err), axis=0)

  return {
    "error": err,
    "L2": L2,
    "L2_spatial": L2_spatial,
    "Linfty_spatial": Linfty_spatial
  }



#



