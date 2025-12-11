# custom_nn.py
# Mathew Titus, October 2025
# Sunstrand Technical Consulting
# 
# This module defines a custom neural network architecture using TensorFlow.
# 
# The custom objects need `get_config` and `from_config` methods defined to 
# override the superclass behavior to allow Keras to serialize the model.
# This implementation allows for manual calls to collect the model structure
# and the model weights:
#   net_con = net.get_config()
#   net_wgt = net.get_weights()
# Then one can recreate the model by calling 
#   net = NetworkModel.from_config(net_con)
# and replacing the default weights with a net.set_weights(net_wgt) call.
# 
# 
# 
# Define the data structure for constructing subnetworks in `config.toml` file.
# 
# {
#   'site_code': {
#       'input_vars': ['site_code_m0', 'site_code_m1', 'site_code_m2', ...],
#       'output_vars': ['site_code_pred'],
#       'adjacencies': ['other_site_code1', 'other_site_code2'],
#       'layers': {'type': 'Dense', 'units': [16, 8, 4], 'activation': 'relu'}
#   }
# }
# 
# Subnetworks are then concatenated and fed to a FFN for final predictions.
# {
#   'layers': [
#       {'type': 'Dense', 'units': [64, 32, 10], 'activation': 'relu'}
#   ]
# }
# 
####################################################################################

import os
import datetime as dt
import toml
import argparse
import numpy as np
import tensorflow as tf
from functools import reduce
import pandas as pd

args = argparse.ArgumentParser(description="Custom Neural Network Module")
args.add_argument("--config", type=str, default="config.toml", help="Path to the configuration TOML file.")


class WeightedMSE(tf.keras.Loss):
  def __init__(self, weights, variable_names=[], name="weighted_mse"):
    super(WeightedMSE, self).__init__(name=name)
    self.weights = weights

  def call(self, y_true, y_pred):
      return tf.reduce_mean(self.weights * tf.square(y_pred - y_true), axis=-1)


class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(CustomDenseLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='weights'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='biases'
        )

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            return self.activation(z)
        return z


class Selector(tf.keras.Initializer):
  def __init__(self, variable_selection):
    self.variable_selection = variable_selection

  def __call__(self, shape, dtype=None, **kwargs):
    x = np.eye(shape[0], dtype=dtype)
    x = x[:,self.variable_selection]
    return x


class VariableSelectionLayer(tf.keras.layers.Layer):
  def __init__(self, var_list, total_vars):
    super(VariableSelectionLayer, self).__init__()
    self.var_list = var_list
    self.total_vars = total_vars
    self.units = 1 # since we are just selecting variables: vector in, vector out

  def build(self, input_shape):
    # print("Building VSL, input_shape: ", input_shape)
    # for each input variable (total_vars), check if it belongs to var_list
    selected_vars = list(
      map(
        lambda var: reduce(
          lambda x, y: var.find(y)>=0 or x,
          self.var_list,
          False
        ),
        self.total_vars
      )
    )
    # self.selected_vars = selected_vars

    selector = Selector(selected_vars)

    self.w = self.add_weight(
            shape=(input_shape, sum(selected_vars)),
            initializer=selector,
            trainable=False,
            name='weights'
        )

    self.b = self.add_weight(
            shape=(self.units, sum(selected_vars)),
            initializer='zeros',
            trainable=False,
            name='biases'
        )

    # self.w = np.eye(input_shape, dtype=np.float32)[:, self.selected_vars]
    # self.b = np.zeros((self.units, sum(selected_vars)))

  def call(self, inputs):
    # print("VSL inputs: ", inputs)
    selection = tf.matmul(inputs, self.w) + self.b
    return selection


class Subnetwork(tf.keras.layers.Layer):
  def __init__(self, 
      network_config: dict, 
      total_vars: list, 
      name: str = "", 
      trainable: bool = True,
      dtype: str = 'float32'
    ):
    
    super(Subnetwork, self).__init__(name=name, trainable=trainable, dtype=dtype)
    self.name = name
    self.layers = []
    self.total_vars = total_vars
    self.config = network_config
    self.name = self.config['name']
    self.activation = self.config['layers']['activation']

  def build(self, input_shape):
    # Perform variable selection
    x = VariableSelectionLayer(var_list=self.config['input_vars'], total_vars=self.total_vars)
    x.build(input_shape[-1])
    self.layers.append(x)

    # Create DNN
    layers = []
    for units in self.config['layers']['units']:
      layer = CustomDenseLayer(units=units, activation=self.activation)
      self.layers.append(layer)

    # build each layer, using shape from previous
    for _ind in range(1, len(self.layers)):
      layer = self.layers[_ind]
      layer.build(self.layers[_ind-1].b.shape)

  def call(self, inputs):
    # print("Calling subnetwork:", self.name)
    x = inputs
    for layer in self.layers:
      x = layer(x)
    return x

  def get_config(self):
    config = super().get_config()
    config.update({
        "network_config": self.config,
        "total_vars": self.total_vars,
    })
    return config

  @classmethod
  def from_config(cls, config):
    # base_config
    return cls(**config)


@tf.keras.utils.register_keras_serializable(
  # package arg: The package that this class belongs to. This is used for the key (which is "package>name") to idenfify the class. Note that this is the first argument passed into the decorator.
  # name arg: The name to serialize this class under in this package. If not provided or None, the class' name will be used (note that this is the case when the decorator is used with only one argument, which becomes the package).
  package='custom_nn', name='NetworkModel'
)
class NetworkModel(tf.keras.Model):
  def __init__(self, 
      network_config: dict, 
      total_vars: list, 
      name: str = "", 
      trainable: bool = True,
      dtype: str = 'float32'
    ):
    super(NetworkModel, self).__init__(name=name, trainable=trainable, dtype=dtype)
    self.network_config = network_config
    self.total_vars = total_vars if isinstance(total_vars, list) else total_vars.tolist()
    self.topology = []
    self.outputs = []

  def build(self, input_shape):
    for subnetwork in self.network_config:
      # print("Adding subnetwork to topology:", subnetwork)
      sn = Subnetwork(subnetwork, self.total_vars)
      sn.build(input_shape)
      self.topology.append(sn)
      self.outputs.extend(subnetwork['output_vars'])

  def call(self, inputs):
    # x = tf.keras.Input(shape=(None, len(input_ntwk_vars)))
    called_subnetworks = [sn(inputs) for sn in self.topology]
    # print("Final topology:", self.topology)
    self.network = tf.keras.layers.Concatenate(axis=-1)(called_subnetworks)
    return self.network

  def get_config(self):
    config = super().get_config()
    config.update({
        "network_config": self.network_config,
        "total_vars": self.total_vars,
    })
    return config

  @classmethod
  def from_config(cls, config):
    # base_config
    inst = cls(**config)
    inst.__init__(config['network_config'], config['total_vars'])
    return inst


def train_subnetwork(element_index, config, total_vars, X=np.array([]), Y=np.array([])):
  '''
  Trains a single subnetwork from the overall NetworkModel.
  Training data is loaded from ./tmp folder if not passed in.

  NB: Data should be saved as np.ndarray objects with shape (time_steps, 1, num_features).
  '''
  # enable tensorboard
  import utils as ut
  log_dir = ut.get_root() + "logs/fit/" + ut.now_string()
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  if X.size == 0 or Y.size == 0:
    # load training data
    X = np.load("./tmp/training_data.npy")
    Y = np.load("./tmp/training_target.npy")
  
  assert (X.size > 0) and (Y.size > 0), "Training data X and Y must be provided or loadable from ./tmp/"

  # parse config info for subnetwork
  subconfig = dict(**config)
  subconfig['network'] = [subconfig['network'][element_index]]
  assert len(subconfig['network']) == 1, "Subnetwork config should contain only one element."

  # get name of singular predictand
  element_name = subconfig['network'][0]['name']

  # build model
  input_layer = tf.keras.Input(shape=(None, len(total_vars)))
  net = NetworkModel(network_config=subconfig['network'], total_vars=list(total_vars), name=element_name)
  predictand = net.outputs

  model = tf.keras.Model(inputs=input_layer, outputs=net(input_layer))

  # define training protocol
  training_params = subconfig['training_parameters']
  # subconfig['optimizer']['learning_rate'] = 0.001

  optimizer = tf.keras.optimizers.Adadelta(
    learning_rate=subconfig['topology']['learning_rate'][element_index],
    **subconfig['optimizer']
  )
  
  # model.compile(optimizer=optimizer, loss="mse") # tf.keras.losses.MeanAbsolutePercentageError())
  model.compile(optimizer=optimizer, loss="mape")

  # fit the model
  history = model.fit(
      X,
      Y,
      callbacks=[tensorboard_callback],
      **training_params
    )

  from plotting import plot_samples
  X2 = np.load("./tmp/testing_data.npy")
  Y2 = np.load("./tmp/testing_target.npy")
  Y2 = Y2[:,:,element_index].reshape((-1,1,1))
  t = pd.read_csv("./tmp/testing_times.csv")
  plot_samples(t, Y2, model.predict(X2), num_elements=8, num_series=5)

  input("Review training performance with tensorboard / plotting output.\nPress Enter to save subnetwork model; Ctrl-C to abort...\n")

  # extract the weights
  wts = model.layers[1].get_weights()
  comp = net.get_weights()

  # assert wts == comp, "Weights from model and subnetwork do not match!"

  os.makedirs("./tmp/", exist_ok=True)

  # save model structure & weights
  net.save(f"./tmp/subnetwork_{element_index}.keras")
  net.save_weights(f"./tmp/subnetwork_{element_index}.weights.h5")

  return net


def test_subnetwork(model_path, X=np.array([]), Y=np.array([])):
  '''
  WIP
  Tests a single subnetwork from the overall NetworkModel.
  Testing data is loaded from ./tmp folder if not passed in.
  '''
  if X.size == 0 or Y.size == 0:
    # load testing data
    X = np.load("./tmp/testing_data.npy")
    Y = np.load("./tmp/testing_target.npy")

  # build model
  input_layer = tf.keras.Input(shape=(None, len(total_vars)))
  net = NetworkModel(network_config=subconfig['network'], total_vars=list(total_vars), name=element_name)
  predictand = net.outputs

  model = tf.keras.Model(inputs=input_layer, outputs=net(input_layer))

  # evaluate the model
  model.evaluate(X, Y)



#



# # define network structure & variables
# with open(args.config, 'r') as f:
#   config = toml.load(f)


# network_elements = ['33', '22', '11'];
# input_ntwk_vars = ['33_m0', '33_m1', '33_m2', 
#               '22_m0', '22_m1', '22_m2', 
#               '11_m0', '11_m1', '11_m2',
#               'rainfall_m0', 'rainfall_m1', 'rainfall_m2'];
# output_ntwk_vars = ['33_pred', '22_pred', '11_pred'];

# network_config = [
#   {
#     'name': '33',
#     'input_vars': ['33', '22', '11', 'rainfall'],
#     'output_vars': ['33_pred'],
#     # 'adjacencies': ['22', '11'],
#     'layers': {'type': 'Dense', 'units': [8, 4, 1], 'activation': 'relu'}
#   },
#   {
#     'name': '22',
#     'input_vars': ['22', '33', 'rainfall'],
#     'output_vars': ['22_pred'],
#     # 'adjacencies': ['33'],
#     'layers': {'type': 'Dense', 'units': [8, 4, 1], 'activation': 'relu'}
#   },
#   {
#     'name': '11',
#     'input_vars': ['11', '33', 'rainfall'],
#     'output_vars': ['11_pred'],
#     # 'adjacencies': ['33'],
#     'layers': {'type': 'Dense', 'units': [8, 4, 1], 'activation': 'relu'}
#   } 
# ]


