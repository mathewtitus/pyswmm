# train_network_model.py
# Mathew Titus, Sunstrand Technical Consulting
# November, 2025
# 
# Create a `NetworkModel` object as defined in custom_nn.py
# Train, validate, save metadata, and create figures.
# 
################################################################

# imports
import os
import argparse
import toml
import json
import tensorflow as tf
import pandas as pd
import numpy as np

# internal imports
from custom_nn import NetworkModel, WeightedMSE, train_subnetwork
import utils as ut
import data_collector as dc
import plotting as pltswmm

args = argparse.ArgumentParser(description="Custom Neural Network Module")
args.add_argument("--config", type=str, default="config.toml", help="Path to the configuration TOML file.")

current_timestamp = ut.now_timestamp()

# enable tensorboard
log_dir = ut.get_root() + "logs/fit/" + ut.now_string()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if __name__ == "__main__":
  args = args.parse_args()
  print(args)

  with open(args.config, "r") as f:
      config = toml.load(f)


  prep_data=True
  if (prep_data):
    '''runs4training, run4testing, X1, Y1, t1, X2, Y2, t2, all_vars, xdims'''
    
    # select dataset
    template = config['template']  # name of system (topology)
    model_structure = "flow_monitor" # link_only; flow_monitor; meso; full
    run_name = config['run']     # name of scenario

    num_runs = config['num_training_runs']        # number of runs to load for training the model
    num_test_runs = config['num_test_runs']    # number of runs to use in testing
    
    # load data
    runs4training, training_data, run4testing, test_data = dc.get_swmm_data(
      num_runs=num_runs, 
      num_test_runs=num_test_runs,
      data_folder=config['data_folder']
    )
    
    # split into predictors, predictands, and series' time index
    [X1, Y1, t1] = training_data
    [X2, Y2, t2] = test_data

    # info to make tf-friendly `fit` call
    all_vars = X1.columns
    xdims = (-1,1,len(all_vars))


  load_data=False #not prep_data
  if (load_data):
    '''runs4training, run4testing, X1, Y1, t1, X2, Y2, t2, all_vars, xdims'''

    assert False, "Loading data is not yet implemented."

    # load previously saved training & testing data from ./tmp
    X1 = np.load("./tmp/training_data.npy")
    Y1 = np.load("./tmp/training_target.npy")
    X2 = np.load("./tmp/testing_data.npy")
    Y2 = np.load("./tmp/testing_target.npy")
    # X1 = pd.DataFrame(
    #   data=X1np.reshape((-1, X1np.shape[-1]))
    # )
    # Y1 = pd.DataFrame(
    #   data=np.load("./tmp/training_target.npy").reshape((-1, np.load("./tmp/training_target.npy").shape[2])),
    #   columns=dc.get_variable_names(config['template'], predictand=True)
    # )
    # X2 = pd.DataFrame(
    #   data=np.load("./tmp/testing_data.npy").reshape((-1, np.load("./tmp/testing_data.npy").shape[2])),
    #   columns=dc.get_variable_names(config['template'])
    # )
    # Y2 = pd.DataFrame(
    #   data=np.load("./tmp/testing_target.npy").reshape((-1, np.load("./tmp/testing_target.npy").shape[2])),
    #   columns=dc.get_variable_names(config['template'], predictand=True)
    # )
    t2 = pd.read_csv("./tmp/testing_times.csv")

    all_vars = X1.columns
    xdims = (-1,1,len(all_vars))


  # define all predictands
  predictands = []
  for _ in config['network']: predictands.extend(_['output_vars'])

  build_model=False
  if (build_model):
    '''net, model'''

    # build model
    input_layer = tf.keras.Input(shape=(None, len(all_vars)))
    net = NetworkModel(network_config=config['network'], total_vars=list(all_vars))

    model = tf.keras.Model(inputs=input_layer, outputs=net(input_layer))

    model_metadata = dict(
      model_type="NetworkModel",
      config=config,
      training_runs=runs4training.tolist(),
      test_runs=run4testing.tolist(),
      input_vars=X1.columns.to_list(),
      output_vars=Y1.columns.to_list(),
      predictands=predictands,
      dataset_size=X1.shape[0],
      loss=None,
      log_dir=log_dir
    )
  

  # info to make tf-friendly `fit` call
  ydims = (-1,1,len(predictands))


  save_data=False
  if (save_data):
    '''None'''
    # save training & testing data to ./tmp for reuse
    np.save("./tmp/training_data.npy", X1.to_numpy().reshape(xdims))
    np.save("./tmp/training_target.npy", Y1.get(predictands).to_numpy().reshape(ydims))
    np.save("./tmp/testing_data.npy", X2.to_numpy().reshape(xdims))
    np.save("./tmp/testing_target.npy", Y2.get(predictands).to_numpy().reshape(ydims))
    t2.to_csv("./tmp/testing_times.csv", index=False)


  load_model=False #not build_model
  if load_model:
    '''net, model'''
    # load model
    model_file = ut.get_root() + "templates/ws_corrected/3day/models/3day_model_1762839160.keras"
    net = tf.keras.models.load_model(model_file, custom_objects={"NetworkModel": NetworkModel})

    input_layer = tf.keras.Input(shape=(None, len(all_vars)))
    model = tf.keras.Model(inputs=input_layer, outputs=net(input_layer))

    metadata_file = model_file.replace("model", "metadata").replace(".keras", ".json")
    with open(metadata_file, "r") as f:
      model_metadata = json.load(f)

  train_piecewise = True
  if (train_piecewise):
    submodel_indices = np.arange(8).tolist()
    subnetwork_savepath = f"./tmp/full_franken_run_X.weights.h5"
    while len(submodel_indices) > 0:
      _ = submodel_indices[0]
      print(f"Training subnetwork {_}...")
      tnet = train_subnetwork(
        _, 
        config, 
        all_vars, 
        X1.to_numpy().reshape((-1,1,len(all_vars))), 
        Y1.get(predictands[_]).to_numpy().reshape((-1,1,1))
      )
      resp = input("Accept model? (y/n)")
      if resp in ['Y', 'y', 'yes']:
        savepath = subnetwork_savepath.replace("X", str(_))
        tnet.save_weights(savepath)
        with open(savepath.replace(".weights.h5", ".json"), "w") as f:
          json.dump(tnet.get_config(), f, indent=1)
        _ = submodel_indices.pop(0)
      elif resp in ['N', 'n', 'no']:
        continue
      else:
        print(f"Response {resp} unclear. Continuing execution without saving.")

    subnetwork_weight_files = list(map(
      lambda x: subnetwork_savepath.replace("X", str(x)),
      np.arange(8)
    ))
    subnetwork_config_files = list(map(
      lambda x: x.replace(".weights.h5", ".json"),
      subnetwork_weight_files
    ))
    frankenput = dict(
      config=config,
      total_vars=all_vars,
      model_list=subnetwork_config_files,
      weight_list=subnetwork_weight_files
    )

    import frankenstein as franken
    net = franken.model_concat(**frankenput)
    net.compile(
      optimizer=tf.keras.optimizers.Adadelta(), 
      loss = "mape"
    )
    net_loss = net.evaluate(X2, Y2.get(predictands))

    # save full model
    full_weights_savepath = f"./templates/{config['template']}/{config['run']}/models/{config['run']}_model_{current_timestamp}.weights.h5"
    save_path = full_weights_savepath.replace(".weights.h5", ".keras") # defining for figurename saving protocol below
    full_config_savepath = full_weights_savepath.replace(".weights.h5", ".json")

    net.save_weights(full_weights_savepath)
    with open(full_config_savepath, 'w') as f:
      json.dump(
        net.get_config(),
        f,
        indent=1
      )

    # save metadata
    model_metadata = dict(
      model_type="FrankenModel",
      config=config,
      config_path=full_config_savepath,
      weights_path=full_weights_savepath,
      training_runs=runs4training.tolist(),
      test_runs=run4testing.tolist(),
      input_vars=X1.columns.to_list(),
      output_vars=Y1.columns.to_list(),
      predictands=predictands,
      dataset_size=X1.shape[0],
      loss=net_loss,
      log_dir=log_dir
    )

    full_metadata_savepath = full_config_savepath.replace("_model_", "_metadata_")
    with open(full_metadata_savepath, 'w') as f:
      json.dump(model_metadata, f, indent=1)

    # create target/pred combo for plotting
    target = Y2.get(model_metadata['predictands'])

    input_layer = tf.keras.Input(shape=(None, len(all_vars)))
    model = tf.keras.Model(inputs=input_layer, outputs=net(input_layer))
    pred = model.predict(X2.to_numpy().reshape(xdims))
    pred = pd.DataFrame(
      data=pred.reshape((pred.shape[0], pred.shape[2])), 
      columns=model_metadata['predictands']
    )

    # save time series index
    (path2models, ) = ut.populate_paths(template, run_name, ['models'])
    the_times = t2.apply(lambda x: int(x.timestamp()))
    with open(f"{path2models}/{run_name}_times_{current_timestamp}.json", "w") as f:
      f.write(the_times.to_json(indent=1))


  # optimizer = tf.keras.optimizers.Adadelta(
  #   **config['optimizer']
  # ) # SGD(0.1) # SGD(0.01) # Adam(0.01)

  # # custom_loss_weights = config['topology'].get('amplification', tf.constant(1.0))
  # # custom_loss = WeightedMSE(weights=custom_loss_weights, variable_names=predictands)
  # # model.compile(optimizer=optimizer, loss=custom_loss)
  # model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError())
  

  train_model=False
  if train_model:
    '''training_params, history'''

    # train model
    training_params = config['training_parameters']

    history = model.fit(
      X1.to_numpy().reshape(xdims),
      Y1.get(predictands).to_numpy().reshape(ydims),
      callbacks=[tensorboard_callback],
      **training_params
    )


  validate_model=False
  if validate_model:
    # validate model on X2 data (test_runs content)
    pred = model.predict(X2.to_numpy().reshape(xdims))
    pred = pd.DataFrame(
      data=pred.reshape((pred.shape[0], pred.shape[2])), 
      columns=model_metadata['predictands']
    )

    # transform target data
    target = Y2.get(model_metadata['predictands'])

    # create error dataframe
    err = (pred - target).abs()
    svin = model.compute_loss(X2, target, pred)
    print(f"Validation SVIN: {svin._numpy()}")

    # update metadata with score
    model_metadata.update(dict(
      loss=float(svin._numpy()) # numpy.float32 not serializable!
    ))

    # calculate residuals
    (err, L2, L2_spatial, Linfty_spatial) = list(ut.get_loss(target, pred).values())

    # save error info
    (path2perf, ) = ut.populate_paths(template, run_name, ['performance'])

    with open(f"{path2perf}/error_{current_timestamp}.json", "w") as f:
      f.write(err.to_json(indent=1))

    with open(f"{path2perf}/L2_temporal_{current_timestamp}.json", "w") as f:
      json.dump(L2.to_list(), f, indent=1)

    with open(f"{path2perf}/L2_spatial_{current_timestamp}.json", "w") as f:
      f.write(L2_spatial.to_json(indent=1))

    with open(f"{path2perf}/Linfty_spatial_{current_timestamp}.json", "w") as f:
      f.write(Linfty_spatial.to_json(indent=1))
  

  save_model=False
  if save_model:
    # save model
    (path2models, ) = ut.populate_paths(template, run_name, ['models'])
    save_path = f"{path2models}/{run_name}_model_{current_timestamp}.keras"
    model.save(save_path)

    # save metadata
    model_metadata.update(dict(
      model_path=save_path,
      predictands=predictands
    ))

    with open(f"{path2models}/{run_name}_metadata_{current_timestamp}.json", "w") as f:
      json.dump(model_metadata, f, indent=1)

    # save time series' index
    the_times = t2.apply(lambda x: int(x.timestamp()))
    with open(f"{path2models}/{run_name}_times_{current_timestamp}.json", "w") as f:
      f.write(the_times.to_json(indent=1))


  ############ Plot results ############
  (path2figs, ) = ut.populate_paths(template, run_name, ['figures'])
  figure_basename = os.path.basename(save_path).replace("model", "xxx").replace("keras", "png")
  figure_path_template = path2figs + '/' + figure_basename

  plot_history=False #validate_model # without predictions & errors, can't make some plots
  if (plot_history):
    # plot training loss
    figname = figure_path_template.replace("xxx", "loss")
    pltswmm.plot_loss(history, figname)

    # determine unlagged rainfall predictor & plot
    rain_cols = [x for x in X2.columns if x.find("rainfall_m0")>=0]
    rain_data = X2.get(rain_cols[0])

    # prep index of each series
    times = t2.apply(lambda x: int(x.timestamp()))
    rain_data = rain_data.reset_index(drop=True)
    times = times.reset_index(drop=True)
    L2 = L2.reset_index(drop=True)

    # plot rain & system-wide error to see if error is related to wet/dry periods, sudden downpour, etc.
    figname = figure_path_template.replace("xxx", "val")
    test_runs_to_plot = pltswmm.plot_rain_vs_loss(model_metadata, times, rain_data, L2, figname) # collect test run #s for reuse

    # repeat with log-error (NB: plot_rain_vs_loss must be run first)
    figname = figure_path_template.replace("xxx", "val_log")
    test_runs_to_plot = pltswmm.plot_rain_vs_logloss(model_metadata, times, rain_data, L2, figname, test_subset=test_runs_to_plot) # plot same runs as above


  plot_elements=True
  if (plot_elements):
    # plot time series comparisons between SWMM output and surrogate output (pick worst node & link, TODO: pick typical node & link)
    figname = figure_path_template.replace("xxx", "worstcase")
    pltswmm.plot_worst_elmts(target, pred, figname)

    # plot time series comparisons between SWMM output and surrogate output (pick worst node & link, TODO: pick typical node & link)
    figname = figure_path_template.replace("xxx", "bestcase")
    pltswmm.plot_best_elmts(target, pred, figname)

    # plot time series comparisons between SWMM output and surrogate output - random subset of links
    figname = '' #figure_path_template.replace("xxx", "sampling")
    pltswmm.plot_samples(t2, target, pred, num_elements=8, num_series=5)



  # # load model and test its performance
  # load_path = save_path
  # restored_model = tf.keras.models.load_model(load_path, custom_objects={'NetworkModel': NetworkModel})
  # repred = restored_model.predict(X2.to_numpy().reshape(xdims))
  # reerr = Y2.get(predictands[1])[:10].to_numpy() - repred[:10,:,1].flatten()




#






