# learning_tf
# Mathew Titus, Sunstrand Technical Consulting
# June, 2024
# 
# NB: Run from `pyswmm` repo root
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


def make_model(input_shape, hidden1_shape, hidden2_shape, output_shape):
  # define MPL model
  model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(hidden1_shape, activation='relu'),
    tf.keras.layers.Dropout(0.4), # read these docs
    tf.keras.layers.Dense(hidden2_shape, activation='relu'),
    tf.keras.layers.Dense(output_shape)
  ])
  # 
  model.compile(
    loss='mean_absolute_error',
    optimizer=tf.keras.optimizers.Adam(0.001)
  )
  # 
  return model;


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



if __name__=="__main__":
  args = sys.argv
  print(f"args: {args}")

  # define model info
  template = "ws_full"  # name of system (topology)
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


  fig, ax = plt.subplots()
  fig.set_size_inches((10,6))
  plot_loss(history, ax)
  plt.savefig(f"{path2figs}/{run_name}_loss_{current_timestamp}.png")
  # plt.show()
  plt.close()

  # test model's performance
  # calculate residuals
  Y2p = model.predict(X2)
  err = np.abs(Y2p - Y2); # numpy array
  L2 = np.sqrt(np.sum(err**2, axis=1))
  L2_spatial = np.sqrt(np.sum(err**2, axis=0))
  Linfty_spatial = np.max(err, axis=0)

  # save error info
  with open(f"{path2perf}/error_{current_timestamp}.json", "w") as f:
    f.write(err.to_json(indent=1))

  with open(f"{path2perf}/L2_temporal_{current_timestamp}.json", "w") as f:
    json.dump(L2.to_list(), f, indent=1)

  with open(f"{path2perf}/L2_spatial_{current_timestamp}.json", "w") as f:
    f.write(L2_spatial.to_json(indent=1))

  with open(f"{path2perf}/Linfty_spatial_{current_timestamp}.json", "w") as f:
    f.write(Linfty_spatial.to_json(indent=1))

  # determine unlagged rainfall predictor & plot
  rain_cols = [x for x in X2.columns if x.find("rainfall_m0")>=0]
  rain_data = X2.get(rain_cols[0])

  # prep index of each series
  times = t2.apply(lambda x: int(x.timestamp()))
  rain_data = rain_data.reset_index(drop=True)
  times = times.reset_index(drop=True)
  L2 = L2.reset_index(drop=True)

  # initialize charts
  fig, ax = plt.subplots(num_test_runs, 1, sharex=True)
  fig.set_size_inches((10,9))
  # annotate plot
  ax[0].title.set_text(f"Model trained on {X1.shape[0]} data points, validated on a hold-out 48-hr period.")

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

  plt.savefig(f"{path2figs}/{run_name}_val_{current_timestamp}.png")
  # plt.show()
  # TODO: color using goldenrod `color=(218/255, 165/255, 32/255)` and black `color=(0, 0, 0)`

  plt.close()

  # repeat with log-error
  fig, ax = plt.subplots(num_test_runs, 1, sharex=True)
  fig.set_size_inches((10,9))
  # annotate plot
  ax[0].title.set_text(f"Model trained on {X1.shape[0]} data points, validated on a hold-out 48-hr period.")

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

  plt.savefig(f"{path2figs}/{run_name}_val_log_{current_timestamp}.png")
  # TODO: color using goldenrod `color=(218/255, 165/255, 32/255)` and black `color=(0, 0, 0)`

  plt.close()

  # TODO: Create time series comparisons between SWMM output and surrogate output (pick worst node & link, pick typical node & link)
  # find worst node & link
  node_colms = list(filter(lambda x: x.find("invert_depth")>=0, Y2.columns))
  link_colms = list(filter(lambda x: x.find("capacity")>=0, Y2.columns))
  node_err = err.get(node_colms)
  link_err = err.get(link_colms)
  peggiore_nodo_ind = np.argmax(node_err.max())
  peggiore_nesso_ind = np.argmax(node_err.max())
  pegg_node = node_colms[peggiore_nodo_ind]
  pegg_nesso = link_colms[peggiore_nesso_ind]

  # plot worst examples of prediction
  fig, ax = plt.subplots(2,1)
  fig.set_size_inches((10,6))
  ax[0].plot(np.arange(Y2.shape[0]), Y2[pegg_node])
  ylims = ax[0].get_ylim()
  ax[0].plot(np.arange(Y2.shape[0]), Y2p[:, peggiore_nodo_ind])
  ax[0].set_ylim(ylims)
  ax[1].plot(np.arange(Y2.shape[0]), Y2[pegg_nesso])
  ylims = ax[1].get_ylim()
  ax[1].plot(np.arange(Y2.shape[0]), Y2p[:, peggiore_nesso_ind])
  ax[1].set_ylim(ylims)

  plt.savefig(f"{path2figs}/{run_name}_worstcase_{current_timestamp}.png")
  plt.show()
  plt.close()

  # plt.scatter(Y2[pegg_node], Y2p[:, peggiore_nodo_ind])
  # plt.title("Simulation vs. Prediction for\nWorst Performing Node")
  # plt.show()

  # plt.scatter(Y2[pegg_nesso], Y2p[:, peggiore_nesso_ind])
  # plt.title("Simulation vs. Prediction for\nWorst Performing Link")
  # plt.show()


