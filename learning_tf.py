# learning_tf
# Mathew Titus, Sunstrand Technical Consulting
# June, 2024
# 
# NB: Run from `pyswmm` repo root
# NB: Update SWMM output filepath in `populate_paths` to select desired
#     JSON data family (# steps, cumulative vars, etc.)
# 
# TODO: Read keras.layers.Dropout documentation (confirm structure from `make_model`)
# 
################################################################################

import tensorflow as tf
import os
import sys
import json
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from simulation_extractor import extraction, tf_prep
from swmm_utils import temp_nodes, temp_links, perm_nodes, perm_links, meso_nodes, meso_links
from utils import get_data
import plotting as pltswmm


log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def filter_vars(X1, X2, Y1, Y2, model_structure):
  '''
  Implement variable filtering, converting SWMM output DataFrame
  into data for a neural emulator with given `model_structure`.
  '''
  if model_structure == "full":
    # keep all variables
    def filtering(x):
      return True
  elif model_structure == "meso":
    # filter to the few targeted variables in `meso_network`
    def filtering(suffixed_colm):
      # drop suffix
      stripped_colm = "_".join(suffixed_colm.split("_")[:-1])
      if suffixed_colm.find("rainfall") >= 0:
        return True
      else:
        # check for an exact match, since link names contain the names of other elements
        if (stripped_colm+"_capacity" in meso_network)|(stripped_colm+"_invert_depth" in meso_network):
          return True          
        # if none of the meso_network names are found in the column name, then
        return False
    raise Exception(f"Not implemented ({model_structure})") # TODO
  elif model_structure == "link_only":
    # filter out all invert_depth variables, leaving link capacities and rainfall
    def filtering(suffixed_colm):
      return suffixed_colm.find("invert") <= 0
  elif model_structure == "flow_monitor":
    # filter to the few targeted variables in `monitor_network`
    def filtering(suffixed_colm):
      # drop variable name (invert_depth / capacity) and lag suffix (e.g. m0, m1, ...)
      stripped_colm = "_".join(suffixed_colm.split("_")[:-2])
      print(stripped_colm)
      if suffixed_colm.find("rainfall") >= 0:
        return True
      else:
        # check for an exact match, since link names contain the names of other elements
        if (stripped_colm in monitor_network)|(stripped_colm in monitor_network):
          return True          
        # if none of the monitor_network names are found in the column name, then
        return False
    # raise Exception(f"Not implemented ({model_structure})") # TODO
  else:
    raise Exception(f"Not implemented ({model_structure})") # TODO

  # subset variables
  X1 = X1[[x for x in X1.columns if filtering(x)]]
  Y1 = Y1[[x for x in Y1.columns if filtering(x)]]
  X2 = X2[[x for x in X2.columns if filtering(x)]]
  Y2 = Y2[[x for x in Y2.columns if filtering(x)]]

  return (X1, X2, Y1, Y2)


def get_topology(topo_defn):
  '''
  Generate a topology definition structure from summary parameters (`topo_defn` dict).
  Parameters:
    family:str, defines overall structure. Choose from trapezoid, vae, adj_nw (TODO), ...
    input_size: length of data input to first layer
    output_size: length of data output by model (# predictands)
    hidden_layers: list of intermediate layer sizes [trapezoid]
      or in the input - intermediate - latent - intermediate - out layer sizes [vae]
    latent_size: latent layer size (# normal vars) [vae]
  '''
  if topo_defn['family'] == "trapezoid":
    model = make_model(topo_defn['input_size'], topo_defn['hidden_layers'][0], topo_defn['hidden_layers'][1], topo_defn['output_size'])
  elif topo_defn['family'] == "vae":
    import vae
    model = vae.VariationalAutoEncoder(original_dim=topo_defn['input_size'], intermediate_dim=topo_defn['hidden_layers'], latent_dim=topo_defn['latent_size'])
  elif topo_defn['family'] == "adj_nw":
    raise Exception(f"adj_nw not yet implemented.")
  else:
    raise Exception(f"not yet implemented.")
  return model


def make_model(input_shape, hidden1_shape, hidden2_shape, output_shape, opt=tf.keras.optimizers.Adam(0.001)):
  # define MPL model
  model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(hidden1_shape, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(hidden2_shape, activation='relu'),
    tf.keras.layers.Dense(output_shape)
  ])
  # 
  model.compile(
    loss='mean_absolute_error',
    optimizer=tf.keras.optimizers.Adam(0.001)
  )
  # 
  return model


def load_times(time_file):
  """
  Load the given file path using the `json` library's `load` method.
  Convert to a DataFrame and return.
  """
  with open(time_file, "r") as f:
    times = json.load(f)
  # 
  time_df = pd.DataFrame.from_dict(times, orient="index", columns=["time"])
  return time_df;


if __name__=="__main__":
  args = sys.argv
  print(f"args: {args}")

  # define model info
  template = "ws_corrected"  # name of system (topology)
  run_name = "3day"     # name of scenario
  num_runs = 80         # number of runs to load for training the model
  num_test_runs = 5     # number of runs to use in testing

  # setup file paths
  path2runs = f"templates/{template}/{run_name}/outputs"
  path2models = f"templates/{template}/{run_name}/models"
  path2figs = f"templates/{template}/{run_name}/figures"
  path2perf = f"templates/{template}/{run_name}/performance"
  for _path in [path2runs, path2models, path2figs, path2perf]:
    if not os.path.exists(_path):
      os.makedirs(_path)

  # collect run list (as List[int]), i.e. inventory the processed JSON data files
  run_list = os.listdir(path2runs)
  run_list = [int(x.rstrip(".json")) for x in run_list if (x[-5:]==".json")]


  # ######## TODO: REMOVE
  # current_timestamp = "1722322971"
  # model = tf.keras.models.load_model(f"templates/ws_full/3day/models/3day_model_{current_timestamp}.keras")
  # with open(f"templates/ws_full/3day/models/3day_metadata_{current_timestamp}.json", "r") as f:
  #   model_metadata = json.load(f)
  # runs4training = model_metadata['training_runs']
  # run4testing = model_metadata['test_runs']
  # #####################


  # define training & test data
  runs4training = np.random.choice(run_list, num_runs)
  run4testing = np.random.choice(np.setdiff1d(run_list, runs4training), num_test_runs)

  training_data = get_data(path2runs, runs4training)
  test_data = get_data(path2runs, run4testing)

  [X1, Y1, t1] = training_data
  [X2, Y2, t2] = test_data

  input("")
  assert ((X1.columns == X2.columns).all()), "Input columns don't match between training & test sets.";
  assert ((Y1.columns == Y2.columns).all()), "Output columns don't match between training & test sets.";

  # define NN topology
  input_shape = X1.shape[1]
  output_shape = Y1.shape[1]
  hidden_shapes = np.ceil(
    np.exp(
      np.linspace(np.log(output_shape), np.log(input_shape), 4)
    )
  )[1:3]

  # set up model
  model = make_model(X1.shape[1], int(hidden_shapes[1]), int(hidden_shapes[0]), Y1.shape[1])

  # fit model
  history = model.fit(
    X1,
    Y1,
    epochs=75,
    # Suppress logging.
    verbose=2,
    # Calculate validation results on [some]% of the training data.
    validation_split = 0.4
  )

  # save model
  current_timestamp = str(dt.datetime.now().timestamp()).split('.')[0]
  model.save(f"{path2models}/{run_name}_model_{current_timestamp}.keras")

  # save metadata
  model_metadata = dict(
    training_runs=runs4training.tolist(),
    test_runs=run4testing.tolist(),
    model_path=f"{path2models}/{run_name}_model_{current_timestamp}.keras",
    input_vars=X1.columns.to_list(),
    output_vars=Y1.columns.to_list()
  )

  with open(f"{path2models}/{run_name}_metadata_{current_timestamp}.json", "w") as f:
    json.dump(model_metadata, f, indent=1)

  # save time series' index
  the_times = t2.apply(lambda x: int(x.timestamp()))
  with open(f"{path2models}/{run_name}_times_{current_timestamp}.json", "w") as f:
    f.write(the_times.to_json(indent=1))

  # plot loss
  pltswmm.plot_loss()


def get_loss(Y2, Y2p, path2perf, current_timestamp):
  # calculate residuals
  (err, L2, L2_spatial, Linfty_spatial) = list(ut.get_loss(Y2, Y2p).values)
  # err = Y2p - Y2 # signed numpy array
  # L2 = np.sqrt(np.sum(err**2, axis=1))
  # L2_spatial = np.sqrt(np.sum(err**2, axis=0))
  # Linfty_spatial = np.max(np.abs(err), axis=0)

  # save error info
  with open(f"{path2perf}/error_{current_timestamp}.json", "w") as f:
    f.write(err.to_json(indent=1))

  with open(f"{path2perf}/L2_temporal_{current_timestamp}.json", "w") as f:
    json.dump(L2.to_list(), f, indent=1)

  with open(f"{path2perf}/L2_spatial_{current_timestamp}.json", "w") as f:
    f.write(L2_spatial.to_json(indent=1))

  with open(f"{path2perf}/Linfty_spatial_{current_timestamp}.json", "w") as f:
    f.write(Linfty_spatial.to_json(indent=1))
  
  return (err, L2, L2_spatial, Linfty_spatial)


# defining lower resolution networks
meso_network = meso_links
monitor_network = perm_links + temp_links

if __name__=="__main__":
  args = sys.argv
  print(f"args: {args}")

  # define model info
  template = "ws_corrected"  # name of system (topology)
  # template = "ws_simple" # name of system (topology)
  run_name = "3day"     # name of scenario
  num_runs = 200 # 80        # number of runs to load for training the model
  num_test_runs = 50     # number of runs to use in testing
  model_structure = "flow_monitor" # link_only; flow_monitor; meso; full
  split = 0.2           # fraction of data to devote to testing; 1-split goes to training
  epochs = 500           # number of training epochs

  # get dirs
  paths = [
    'outputs_13step', # Update as needed to select different dataset structures
    'models', 
    'figures', 
    'performance'
  ]
  (path2runs, path2models, path2figs, path2perf) = ut.populate_paths(template, run_name, paths)

  # collect run list (as List[int]), i.e. inventory the processed JSON data files
  run_list = os.listdir(path2runs)
  run_list = [int(x.rstrip(".json")) for x in run_list if (x[-5:]==".json")]

  # load or sort runs into training / testing (/ neither) groups
  model2load = ""
  # model2load = "3day_model_1722322971.keras"
  if model2load:
    print(f"Loading model {model2load}...")
    model = tf.keras.models.load_model(f"templates/{template}/{run_name}/models/{model2load}")
    with open(f"templates/{template}/{run_name}/models/{model2load.replace('_model_', '_metadata_').replace('.keras', '.json')}", "r") as f:
      model_metadata = json.load(f)
    runs4training = model_metadata['training_runs']
    run4testing = model_metadata['test_runs']
    num_runs = len(runs4training)
    num_test_runs = len(run4testing)
    model_structure = model_metadata['model_structure']
  else:
    # define training & test data
    runs4training = np.random.choice(run_list, num_runs)
    run4testing = np.random.choice(np.setdiff1d(run_list, runs4training), num_test_runs)

  training_data = get_data(path2runs, runs4training)
  test_data = get_data(path2runs, run4testing)

  [X1, Y1, t1] = training_data
  [X2, Y2, t2] = test_data

  # define relevant variables
  X1, X2, Y1, Y2 = filter_vars(X1, X2, Y1, Y2, model_structure)
  
  assert ((X1.columns == X2.columns).all()), "Input columns don't match between training & test sets.";
  assert ((Y1.columns == Y2.columns).all()), "Output columns don't match between training & test sets.";

  # define NN topology
  input_shape = X1.shape[1]
  output_shape = Y1.shape[1]
  hidden_shapes = np.ceil(
    np.exp(
      np.linspace(np.log(output_shape), np.log(input_shape), 4)
    )
  )[1:3]

  # set up model
  if model2load:
    pass
  else:
    optimizer = tf.keras.optimizers.Adadelta() # SGD(0.1) # SGD(0.01) # Adam(0.01)
    model = make_model(X1.shape[1], int(hidden_shapes[1]), int(hidden_shapes[0]), Y1.shape[1], opt=optimizer)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # fit model & log data
    history = model.fit(
      X1,
      Y1,
      batch_size=8, # default: 32
      epochs=epochs,
      # Suppress logging.
      verbose=2,
      # Calculate validation results on [some]% of the training data.
      # validation_split = split,
      validation_data=(X2, Y2),
      shuffle=True,
      callbacks=[tensorboard_callback]
    )

    # save model
    current_timestamp = str(dt.datetime.now().timestamp()).split('.')[0]
    model.save(f"{path2models}/{run_name}_model_{current_timestamp}.keras")

    # save metadata
    model_metadata = dict(
      training_runs=runs4training.tolist(),
      test_runs=run4testing.tolist(),
      data_path=f"{path2runs}",
      model_path=f"{path2models}/{run_name}_model_{current_timestamp}.keras",
      model_structure=model_structure,
      input_vars=X1.columns.to_list(),
      output_vars=Y1.columns.to_list(),
      timestamp=current_timestamp,
      split=split
    )

    with open(f"{path2models}/{run_name}_metadata_{current_timestamp}.json", "w") as f:
      json.dump(model_metadata, f, indent=1)

    # save time series' index
    the_times = t2.apply(lambda x: int(x.timestamp()))
    with open(f"{path2models}/{run_name}_times_{current_timestamp}.json", "w") as f:
      f.write(the_times.to_json(indent=1))

  # test model's performance
  Y2p = model.predict(X2)
  err, L2, L2_spatial, Linfty_spatial = get_loss(Y2, Y2p, path2perf, current_timestamp)

  # determine unlagged rainfall predictor & plot
  rain_cols = [x for x in X2.columns if x.find("rainfall_m0")>=0]
  rain_data = X2.get(rain_cols[0])

  # prep index of each series
  times = t2.apply(lambda x: int(x.timestamp()))
  rain_data = rain_data.reset_index(drop=True)
  times = times.reset_index(drop=True)
  L2 = L2.reset_index(drop=True)

  ############ Plot results ############

  # plot historical loss (training / val)
  figname = f"{path2figs}/{run_name}_loss_{current_timestamp}.png"
  plot_loss_wrapper(history, figname)

  # plot rain & system-wide error to see if error is related to wet/dry periods, sudden downpour, etc.
  figname = f"{path2figs}/{run_name}_val_{current_timestamp}.png"
  test_indices, labeled_run = plot_rain_vs_loss(num_test_runs, X1.shape[0], times, figname)

  # repeat with log-error (NB: plot_rain_vs_loss must be run first)
  figname = f"{path2figs}/{run_name}_val_log_{current_timestamp}.png"
  plot_rain_vs_logloss(num_test_runs, X1.shape[0], test_indices, labeled_run, figname)

  # plot time series comparisons between SWMM output and surrogate output (pick worst node & link, TODO: pick typical node & link)
  figname = f"{path2figs}/{run_name}_worstcase_{current_timestamp}.png"
  plot_worst_elmts(Y2, Y2p, figname)

  # plot time series comparisons between SWMM output and surrogate output (pick worst node & link, TODO: pick typical node & link)
  figname = f"{path2figs}/{run_name}_bestcase_{current_timestamp}.png"
  plot_best_elmts(Y2, Y2p, figname)
  






#
