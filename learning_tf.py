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

log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def populate_paths(template, run_name):
  # setup file paths
  path2runs = f"templates/{template}/{run_name}/outputs_13step" # Update here
  path2models = f"templates/{template}/{run_name}/models"
  path2figs = f"templates/{template}/{run_name}/figures"
  path2perf = f"templates/{template}/{run_name}/performance"
  for _path in [path2runs, path2models, path2figs, path2perf]:
    if not os.path.exists(_path):
      os.makedirs(_path)
  
  return (path2runs, path2models, path2figs, path2perf)


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
      datum = pd.DataFrame(new_json)
      data.append(datum)
    except:
      failures.append(run_num)
  # 
  if len(data) > 0:
    data = pd.concat(data)
  else:
    print(f"Error: No data collected in `get_data`. Path={path}, run_list={run_list}.")
    return None
  # 
  if len(failures) > 0:
    print(f"Error in `get_data`: run_list included {failures} but these files failed to load.")
  #
  # validate predictand presence before processing
  predictand_cols = list(filter(lambda x: x.find("_pred")>=0, data.columns))
  assert len(predictand_cols) > 0, f"No predictands found in column list."
  # 
  # NB: unclear what is happening to our timestamps when they're jsonified...
  # this is a rough conversion, adding 16 hours due to timezone madness
  times = data.time.apply(lambda x: dt.datetime.fromtimestamp(x/1000))\
    .reset_index(drop=True)
  df = data.drop(columns="time")\
    .reset_index(drop=True)
  # 
  # assemble predictors & predictands
  X = df.get(np.setdiff1d(df.columns, predictand_cols))
  Y = df.get(predictand_cols)
  # 
  return X, Y, times;


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


def plot_loss(history, ax=None):
  if ax:
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error [$t + \Delta t$]')
    ax.legend()
    ax.grid(True)
  else:
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [$t + \Delta t$]')
    plt.legend()
    plt.grid(True)


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


def subdivide_time_series(times):
  """
  Given a time series composed of k repetitions of a time series with
  distinct entries, returns a dictionary of lists, each list indexing
  a unique copy of the distinct-entry time series.
  Fails if the originating time series does not have a constant k number
  of repetitions of each unique entry.
  """
  subseries = {}
  entries = times.unique()
  # 
  for (ind, ent) in enumerate(entries):
    les_entries = np.where(times==ent)[0]
    subseries[ind] = les_entries
  # calculate 
  len_set = set(map(
    lambda x: len(x),
    subseries.values()
  ))
  assert len(len_set) == 1, f"subdivide_time_series served an anomaly; length set is {len_set}"
  # 
  return subseries;


def plot_loss_wrapper(history, figname):
  '''
  Plot the training loss history (train & val data sets).
  '''
  fig, ax = plt.subplots()
  fig.set_size_inches((10,6))
  plot_loss(history, ax)
  plt.savefig(figname)
  # plt.show()
  plt.close()


def get_loss(Y2, Y2p, path2perf, current_timestamp):
  # calculate residuals
  err = Y2p - Y2 # signed numpy array
  L2 = np.sqrt(np.sum(err**2, axis=1))
  L2_spatial = np.sqrt(np.sum(err**2, axis=0))
  Linfty_spatial = np.max(np.abs(err), axis=0)

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


def plot_rain_vs_loss(num_test_runs, dataset_size, times, figname):
  # initialize charts
  fig, ax = plt.subplots(num_test_runs, 1, sharex=True)
  fig.set_size_inches((10,9))
  # annotate plot
  ax[0].title.set_text(f"Model trained on {dataset_size} data points, validated on a hold-out 48-hr period.")

  # plot each test run's error
  time_index_dict = subdivide_time_series(times)
  test_indices = {}
  for test in range(num_test_runs):
    # parse time_index_dict
    test_indices[test] = []
    for ind in range(len(time_index_dict)):
      test_indices[test].append(time_index_dict[ind][test])
    # collect data from concatenated test data
    time_subseries = t2.iloc[test_indices[test]]
    rain_subseries = rain_data.iloc[test_indices[test]]
    L2_subseries = L2.iloc[test_indices[test]]
    # plot
    ax[test].plot(time_subseries, rain_subseries)
    ax2 = ax[test].twinx()
    ax2.plot(time_subseries, L2_subseries, color="orange")
    # ax.legend()

  # make y-labels
  labeled_run = int(np.floor(num_test_runs/2))

  ax[labeled_run].yaxis.set_label_text("Rainfall")
  ax2 = ax[labeled_run].twinx()
  ax2.yaxis.set_label_text("Network-wide error (L2)")
  ax2.yaxis.set_ticklabels([])

  plt.savefig(figname)
  # plt.show()
  # TODO: color using goldenrod `color=(218/255, 165/255, 32/255)` and black `color=(0, 0, 0)`

  plt.close()

  # return indices for later plots
  return (test_indices, labeled_run)


def plot_rain_vs_logloss(num_test_runs, dataset_size, test_indices, labeled_run, figname):
  fig, ax = plt.subplots(num_test_runs, 1, sharex=True)
  fig.set_size_inches((10,9))
  # annotate plot
  ax[0].title.set_text(f"Model trained on {dataset_size} data points, validated on a hold-out 48-hr period.")

  # plot each test run's error
  # reuse time_index_dict for this one
  for test in range(num_test_runs):
    # collect data from concatenated test data
    time_subseries = t2.iloc[test_indices[test]]
    rain_subseries = rain_data.iloc[test_indices[test]]
    L2_subseries = L2.iloc[test_indices[test]].apply(np.log)
    # plot
    ax[test].plot(time_subseries, rain_subseries)
    ax2 = ax[test].twinx()
    ax2.plot(time_subseries, L2_subseries, color="red")

  # make y-labels
  ax[labeled_run].yaxis.set_label_text("Rainfall")
  ax2 = ax[labeled_run].twinx()
  ax2.yaxis.set_label_text("Network-wide error (log-L2)")
  ax2.yaxis.set_ticklabels([])

  plt.savefig(figname)
  # TODO: color using goldenrod `color=(218/255, 165/255, 32/255)` and black `color=(0, 0, 0)`

  plt.close()


def plot_worst_elmts(Y2, Y2p, figname):
  err = Y2p - Y2

  fig, ax = plt.subplots(2,1)
  fig.set_size_inches((10,6))
  plt.title("Worst performing node & link on test data.")

  node_colms = list(filter(lambda x: x.find("invert_depth")>=0, Y2.columns))
  link_colms = list(filter(lambda x: x.find("capacity")>=0, Y2.columns))
  
  if len(node_colms) > 0:
    # find worst node
    node_err = err.get(node_colms)
    peggiore_nodo_ind = np.argmax(node_err.max())
    pegg_node = node_colms[peggiore_nodo_ind]
    
    # plot worst examples of prediction
    ax[0].plot(np.arange(Y2.shape[0]), Y2[pegg_node])
    ylims = ax[0].get_ylim()
    ax[0].plot(np.arange(Y2.shape[0]), Y2p[:, peggiore_nodo_ind])
    ax[0].set_ylim(ylims)
    ax[0].legend(["Actual", "Predicted"])
  if len(link_colms) > 0:
    # find worst link
    link_err = err.get(link_colms)
    peggiore_nesso_ind = np.argmax(link_err.max())
    pegg_nesso = link_colms[peggiore_nesso_ind]

    # plot worst examples of prediction
    ax[1].plot(np.arange(Y2.shape[0]), Y2[pegg_nesso])
    ylims = ax[1].get_ylim()
    ax[1].plot(np.arange(Y2.shape[0]), Y2p[:, peggiore_nesso_ind])
    ax[1].set_ylim(ylims)
    ax[1].legend(["Actual", "Predicted"])

  plt.savefig(figname)
  # plt.show()
  plt.close()

  # plt.scatter(Y2[pegg_node], Y2p[:, peggiore_nodo_ind])
  # plt.title("Simulation vs. Prediction for\nWorst Performing Node")
  # plt.show()

  # plt.scatter(Y2[pegg_nesso], Y2p[:, peggiore_nesso_ind])
  # plt.title("Simulation vs. Prediction for\nWorst Performing Link")
  # plt.show()


def plot_best_elmts(Y2, Y2p, figname):
  err = Y2p - Y2

  fig, ax = plt.subplots(2,1)
  fig.set_size_inches((10,6))
  plt.title("Best performing node & link on test data.")

  node_colms = list(filter(lambda x: x.find("invert_depth")>=0, Y2.columns))
  link_colms = list(filter(lambda x: x.find("capacity")>=0, Y2.columns))
  
  if len(node_colms) > 0:
    # find worst node
    node_err = err.get(node_colms)
    maggiore_nodo_ind = np.argmax(node_err.max())
    magg_node = node_colms[maggiore_nodo_ind]
    
    # plot worst examples of prediction
    ax[0].plot(np.arange(Y2.shape[0]), Y2[magg_node])
    ylims = ax[0].get_ylim()
    ax[0].plot(np.arange(Y2.shape[0]), Y2p[:, maggiore_nodo_ind])
    ax[0].set_ylim(ylims)
    ax[0].legend(["Actual", "Predicted"])
  if len(link_colms) > 0:
    # find worst link
    link_err = err.get(link_colms)
    maggiore_nesso_ind = np.argmax(link_err.max())
    magg_nesso = link_colms[maggiore_nesso_ind]

    # plot worst examples of prediction
    ax[1].plot(np.arange(Y2.shape[0]), Y2[magg_nesso])
    ylims = ax[1].get_ylim()
    ax[1].plot(np.arange(Y2.shape[0]), Y2p[:, maggiore_nesso_ind])
    ax[1].set_ylim(ylims)
    ax[1].legend(["Actual", "Predicted"])

  plt.savefig(figname)
  # plt.show()
  plt.close()

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

  # create dirs
  # NB: update this method to target different directories (e.g. if you want a different extraction method - 3 hour history, etc.)
  (path2runs, path2models, path2figs, path2perf) = populate_paths(template, run_name)

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
