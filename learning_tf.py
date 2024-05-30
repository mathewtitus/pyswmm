

import tensorflow as tf
import os
import json
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from simulation_extractor import extraction, tf_prep

# define model info
template = "demo_system"
run_name = "3day"
run_list = os.listdir(f"templates/{template}/outputs/{run_name}/")
runs4training = np.random.choice([int(x.rstrip(".json")) for x in run_list if x.find(".json")>=0], 30)


# collect JSON data & convert to DataFrame
data = []
for run_num in runs4training:
  json_path = f"templates/{template}/outputs/{run_name}/{run_num}.json"
  with open(json_path, 'r') as f:
    new_json = json.load(f)
  datum = pd.DataFrame(new_json)
  data.append(datum)

data = pd.concat(data)

# unclear what is happening to our timestamps when they're jsonified...
# this is a rough conversion, adding 16 hours do to timezone madness
times = data.time.apply(lambda x: dt.datetime.fromtimestamp(x/1000))

df = data.drop(columns="time")\
  .reset_index(drop=True)

# assemble predictors & predictands
predictand_cols = list(filter(lambda x: x.find("_pred")>=0, df.columns))
X = df.get(np.setdiff1d(df.columns, predictand_cols))
Y = df.get(predictand_cols)

# # separate into train/test sets
# pass

# # normalize inputs
# ####
# normalizer = tf.keras.layers.Normalization(axis=-1) ## what is axis=-1
# normalizer.adapt(np.array(train_features))
# ####
# horsepower = np.array(train_features['Horsepower'])

# horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
# horsepower_normalizer.adapt(horsepower)
# ####


# define MPL model
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(X.shape[1],)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2), # read these docs
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(Y.shape[1])
])

model.compile(
  loss='mean_absolute_error',
  optimizer=tf.keras.optimizers.Adam(0.001)
)

history = model.fit(
    X,
    Y,
    epochs=100,
    # Suppress logging.
    verbose=2,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.4)


# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)

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

fig, ax = plt.subplots()
fig.set_size_inches((10,6))
plot_loss(history, ax)
plt.savefig(f"./templates/{template}/models/{run_name}_loss_{str(dt.datetime.now().timestamp()).split('.')[0]}.png")
plt.show()

# TODO: quantify error in terms of capacity
pass 


# collect another dataset

val_file = np.random.choice(np.setdiff1d(run_list, [str(x) + ".json" for x in runs4training]))
val_run = f"templates/{template}/outputs/{run_name}/{val_file}"
with open(val_run, 'r') as f:
  val_data = json.load(f)

df2 = pd.DataFrame(val_data)
times2 = df2.time.apply(lambda x: dt.datetime.fromtimestamp(x/1000))

df2 = df2.drop(columns="time")\
  .reset_index(drop=True)

predictand_cols = list(filter(lambda x: x.find("_pred")>=0, df2.columns))
X2 = df2.get(np.setdiff1d(df2.columns, predictand_cols))
Y2 = df2.get(predictand_cols)

# test model's performance


# calculate residuals
Y2p = model.predict(X2)
err = Y2p - Y2; # numpy array
L2 = np.sqrt(np.sum(err**2, axis=1))

# initialize chart
fig, ax = plt.subplots()

# plot error
ax.plot(times2, L2, label="error")

# get rainfall & plot
# ax2 = ax.twinx()
# ax2.plot(X2.S1_rainfall_m0, color='orange', label="rain")
ax.plot(times2, X2.S1_rainfall_m0, label="rain")

ax.legend()
plt.title(f"Model trained on {X.shape[0]} data points, validated on a hold-out 72-hr period.")
fig.set_size_inches((8,6))
plt.savefig(f"./templates/{template}/models/{run_name}_val_{str(dt.datetime.now().timestamp()).split('.')[0]}.png")
plt.show()










# plt.plot(X.S1_rainfall_m0)
# plt.plot(X2.S1_rainfall_m0)
# plt.show()


# for col in X.columns:
#   plt.plot(times, X[col])
#   plt.plot(times, X2[col])
#   plt.title(col)
#   plt.show()


