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





# if __name__=="__main__":
#   args = sys.argv()
#   print(args)

# define model info
template = "ws_full"  # name of system (topology)
run_name = "3day"     # name of scenario
num_runs = 70         # number of runs to load for training the model
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
run_list = [int(x.rstrip(".json")) for x in run_list if x.find(".json")>=0]

# define training & test data
runs4training = np.random.choice(run_list, num_runs)
run4testing = np.random.choice(np.setdiff1d(run_list, runs4training), num_test_runs)

training_data = get_data(path2runs, runs4training)
test_data = get_data(path2runs, run4testing)

[X1, Y1, t1] = training_data
[X2, Y2, t2] = test_data

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
  epochs=100,
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
  in_vars=X1.columns.tolist(),
  out_vars=Y1.columns.tolist()
)

with open(f"{path2models}/{run_name}_metadata_{current_timestamp}.json", "w") as f:
  json.dump(model_metadata, f, indent=1)


fig, ax = plt.subplots()
fig.set_size_inches((10,6))
plot_loss(history, ax)
plt.savefig(f"{path2figs}/{run_name}_loss_{current_timestamp}.png")
plt.show()

# TODO: quantify error in terms of capacity
pass 


# test model's performance


# calculate residuals
Y2p = model.predict(X2)
err = Y2p - Y2; # numpy array
L2 = np.sqrt(np.sum(err**2, axis=1))
L2_spatial = np.sqrt(np.sum(err**2, axis=0))

# save error info
with open(f"{path2perf}/error_{current_timestamp}.json", "w") as f:
  json.dump(err.to_numpy(), f, indent=1) # TODO: Test

with open(f"{path2perf}/L2_temporal_{current_timestamp}.json", "w") as f:
  json.dump(L2.to_list(), f, indent=1)

with open(f"{path2perf}/L2_spatial_{current_timestamp}.json", "w") as f:
  json.dump(L2_spatial.to_json(), f, indent=1)

# initialize chart
fig, ax = plt.subplots()

# plot error
# TODO: save times as ??? in ???
# TODO: collect t2 from ??? and break up by simulation, plotting each separately (no crossing lines from t_final to t_init)
# TODO: color using goldenrod `color=(218/255, 165/255, 32/255)` and black `color=(0, 0, 0)`
ax.plot(t2, L2, label="error")

# determine unlagged rainfall predictor & plot
rain_cols = [x for x in X2.columns if x.find("rainfall_m0")>=0]
ax2 = ax.twinx()
ax2.plot(t2, X2.get(rain_cols[0]), color='orange', label="rain")

# annotate plot
ax.legend()
plt.title(f"Model trained on {X1.shape[0]} data points, validated on a hold-out 48-hr period.")
fig.set_size_inches((10,6))
plt.savefig(f"{path2figs}/{run_name}_val_{current_timestamp}.png")
plt.show()

# TODO: Create time series comparisons between SWMM output and surrogate output (pick worst node & link, pick typical node & link)



