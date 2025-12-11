# frankenstein.py
# Mathew Titus, November 2025
# Sunstrand Technical Consulting
#
# This module uses elements from custom_nn.py and train_network_model.py
# to piece together a single TF model from a collection of trained subnetworks.
# 
# TODO: Make subnetwork weight loading order informed by subnetwork names ordering 
#       in the full NetworkModel.outputs vector.
# 
#####################################################################################

import json
from tensorflow.keras import Input
from custom_nn import NetworkModel


def model_concat(config: dict, total_vars: list, model_list: list, weight_list: list):
  '''
  Instantiates a NetworkModel based on `config` then builds and sets its weights by
  concatenating the weights loaded from each constituent model listed. This organization 
  is based on a NetworkModel being the output of a tf.keras.layers.Concatenate call, with 
  each Subnetwork in the config['network'] definition matching the order of the saved models 
  defined in `model_list` and (with respective ordering) `weight_list`.
  '''
  # define & build network model
  net = NetworkModel(network_config=config['network'], total_vars=total_vars)
  input_layer = Input(shape=(None, len(total_vars)))
  net.build(input_layer.shape)

  assert len(net.layers) == len(model_list), "Mismatch in config file and number of subnetworks provided."

  # load subnetwork as a NetworkModel and record its weights
  full_weight_set = []
  for (_, path) in enumerate(model_list):
    # get config
    with open(path, 'r') as f:
      subconfig = json.load(f)
    # initialize model
    subnet = NetworkModel.from_config(subconfig)
    # build
    subnet.build(input_layer.shape)
    # set weights
    subnet.load_weights(weight_list[_])
    # record for net
    full_weight_set.extend(
      subnet.get_weights()
    )
    
  net.set_weights(full_weight_set)
  return net


#





