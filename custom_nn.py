# custom_nn.py
# Mathew Titus, October 2025
# Sunstrand Technical Consulting
# 
# This module defines a custom neural network architecture using TensorFlow.
# 
####################################################################################

import numpy as np
import tensorflow as tf
from functools import reduce

# Defining the data structure for constructing subnetworks
# `units` sequence defines the number of neurons in each layer of the subnetwork.
# {
#   'site_code': {
#       'input_vars': ['site_code_m0', 'site_code_m1', 'site_code_m2', ...],
#       'output_vars': ['site_code_pred'],
#       'adjacencies': ['other_site_code1', 'other_site_code2'],
#       'layers': {'type': 'Dense', 'units': [16, 8, 4], 'activation': 'relu'}
#   }
# }

# Subnetworks are then concatenated and fed to a FFN for final predictions.
# {
#   'layers': [
#       {'type': 'Dense', 'units': [64, 32, 10], 'activation': 'relu'}
#   ]
# }

# define network structure & variables
network_elements = ['33', '22', '11'];
input_ntwk_vars = ['33_m0', '33_m1', '33_m2', 
              '22_m0', '22_m1', '22_m2', 
              '11_m0', '11_m1', '11_m2',
              'rainfall_m0', 'rainfall_m1', 'rainfall_m2'];
output_ntwk_vars = ['33_pred', '22_pred', '11_pred'];

network_config = [
  {
    'name': '33',
    'input_vars': ['33', '22', '11', 'rainfall'],
    'output_vars': ['33_pred'],
    # 'adjacencies': ['22', '11'],
    'layers': {'type': 'Dense', 'units': [8, 4, 1], 'activation': 'relu'}
  },
  {
    'name': '22',
    'input_vars': ['22', '33', 'rainfall'],
    'output_vars': ['22_pred'],
    # 'adjacencies': ['33'],
    'layers': {'type': 'Dense', 'units': [8, 4, 1], 'activation': 'relu'}
  },
  {
    'name': '11',
    'input_vars': ['11', '33', 'rainfall'],
    'output_vars': ['11_pred'],
    # 'adjacencies': ['33'],
    'layers': {'type': 'Dense', 'units': [8, 4, 1], 'activation': 'relu'}
  } 
]

# pyright: ignore[reportMissingImports]
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


class VariableSelectionLayer(tf.keras.layers.Layer):
  def __init__(self, var_list):
    super(VariableSelectionLayer, self).__init__()
    self.var_list = var_list
    self.units = 1 # since we are just selecting variables: vector in, vector out

  def build(self, input_shape):
    # for each input variable (input_ntwk_vars), check if it belongs to var_list
    selected_vars = list(
      map(
        lambda var: reduce(
          lambda x, y: var.find(y)>=0 or x,
          self.var_list,
          False
        ),
        input_ntwk_vars
      )
    )
    self.selected_vars = selected_vars
    self.w = np.eye(input_shape[-1], dtype=np.float32)[:, self.selected_vars]
    self.b = np.zeros((self.units, sum(selected_vars)))

  def call(self, inputs):
    selection = tf.matmul(inputs, self.w) + self.b
    return selection


class Subnetwork(tf.keras.layers.Layer):
  def __init__(self, network_config: dict):
    # print("Inializing subnetwork:", network_config['name'])
    super(Subnetwork, self).__init__()
    self.layers = []
    self.config = network_config
    self.name = self.config['name']
    self.activation = self.config['layers']['activation']

  def build(self, input_vars):
    # print("Building subnetwork:", self.name)
    # print("Selecting input variables:", input_vars)
    # print("Config is:", self.config)
    # print("Input config variables:", self.config['input_vars'])
    # Perform variable selection
    self.data_filter_layer = VariableSelectionLayer(var_list=self.config['input_vars'])

    # Create DNN
    for units in self.config['layers']['units']:
      layer = CustomDenseLayer(units=units, activation=self.activation)
      self.layers.append(layer)

  def call(self, inputs):
    # print("Calling subnetwork:", self.name)
    x = self.data_filter_layer(inputs)
    for layer in self.layers:
      x = layer(x)
    return x


class NetworkModel(tf.keras.Model):
  def __init__(self, network_config: list):
    super(NetworkModel, self).__init__()
    self.topology = []
    self.outputs = []

  def build(self, input_shape):
    for subnetwork in network_config:
      # print("Adding subnetwork to topology:", subnetwork)
      sn = Subnetwork(subnetwork)
      sn.build(input_shape)
      self.topology.append(sn)
      self.outputs.extend(subnetwork['output_vars'])
    
    

  def call(self, inputs):
    # x = tf.keras.Input(shape=(None, len(input_ntwk_vars)))
    called_subnetworks = [sn(inputs) for sn in self.topology]
    # print("Final topology:", self.topology)
    self.network = tf.keras.layers.Concatenate(axis=-1)(called_subnetworks)
    return self.network


##### SAMPLE USAGE #####


input_layer = tf.keras.Input(shape=(None, len(input_ntwk_vars)))
net = NetworkModel(network_config=network_config)
model = tf.keras.Model(inputs=input_layer, outputs=net(input_layer))
model.compile(optimizer='adam', loss='mse')

exampleData = np.array([
    1.0, 2.0, 3.0, # 33
    4.0, 5.0, 6.0, # 22
    7.0, 8.0, 9.0, # 11
    1.2, 4.2, 0.8  # rainfall
  ]).reshape((1,1,12))

model_output = model.predict(exampleData)
print(model_output)

# adj = np.array([[0, 1, 1],
#                 [1, 0, 0],
#                 [1, 0, 0]])
# adj = adj + np.eye(adj.shape[0])  # Make sure the adjacency matrix has self-loops

# assert np.allclose(adj, adj.T), "Adjacency martrix must be symmetric."




