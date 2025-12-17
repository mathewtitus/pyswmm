import json
import numpy as np
import tensorflow as tf
from custom_nn import NetworkModel

# Collect model data
wts_path = 'templates/ws_corrected/3day/models/3day_model_1763669109.weights.h5'
cfg_path = 'templates/ws_corrected/3day/models/3day_config_1763669109.json'
md_path = 'templates/ws_corrected/3day/models/3day_metadata_1763669109.json'

with open(cfg_path, 'r') as f:
  cfg = json.load(f)

with open(md_path, 'r') as f:
  md = json.load(f)

# Build model
total_vars = cfg['total_vars']
net = NetworkModel.from_config(cfg)
net.load_weights(wts_path)

input_layer = tf.keras.Input(shape=(None, len(total_vars)))
model = tf.keras.Model(inputs=input_layer, outputs=net(input_layer))
model.compile(loss="mape")

# Load data
X = np.load("tmp/testing_data.npy")
Y = np.load("tmp/testing_target.npy")

# Test predictive capability
pred = model.predict(X)
test_score = model.evaluate(X,Y)


