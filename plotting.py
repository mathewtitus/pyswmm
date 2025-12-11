# plotting.py
# Mathew Titus, Sunstrand Technical Consulting
# November, 2025
# 
# Plotting scripts for visualizing model performance.
# 
######################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss(history, figname, ax=None):
  '''
  Given the `history` of a `model.fit` run,
  plot the train/test validations across epochs.
  If an Axis object is provided, the chart will
  be added to it. Then, the figure is saved and closed.
  '''
  if ax:
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error [$t + \Delta t$]')
    ax.legend()
    ax.grid(True)
  else:
    fig, ax = plt.subplots()
    fig.set_size_inches((10,6))

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [$t + \Delta t$]')
    plt.legend()
    plt.grid(True)

  plt.savefig(figname)
  # plt.show()
  plt.close()


def subdivide_time_series(times):
  """
  Supporting function for `plot_rain_vs_loss` function.

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


def plot_rain_vs_loss(metadata, times, rain_data, L2, figname, test_subset=[]):
  # initialize charts
  num_test_runs = len(metadata['test_runs'])
  num_charts = max(min(6, num_test_runs), 2) # at least 2 charts to avoid subscripting issue
  fig, ax = plt.subplots(num_charts, 1, sharex=True)
  fig.set_size_inches((10,9))

  # annotate plot
  ax[0].title.set_text(f"Model trained on {metadata['dataset_size']} data points, validated on a hold-out 48-hr period.")

  # plot each test run's error
  time_index_dict = subdivide_time_series(times)
  test_indices = {}

  # choose at most 6 examples to plot
  if len(test_subset) == 0:
    test_subset = np.random.permutation(
                    np.arange(num_test_runs)
                  )[:min(6,num_test_runs)]
  for ind, test in enumerate(test_subset): # range(num_test_runs):
    print(f"ind: {ind}, test: {test}")
    # parse time_index_dict
    test_indices[test] = []
    for _ind in range(len(time_index_dict)):
      test_indices[test].append(time_index_dict[_ind][test])
    # collect data from concatenated test data
    time_subseries = times.iloc[test_indices[test]] # t2.iloc[test_indices[test]]
    rain_subseries = rain_data.iloc[test_indices[test]]
    L2_subseries = L2.iloc[test_indices[test]]
    # plot
    ax[ind].plot(time_subseries, rain_subseries)
    ax2 = ax[ind].twinx()
    ax2.plot(time_subseries, L2_subseries, color="orange")
    # ax.legend()
    ax[ind].yaxis.set_label_text(test)

  # make y-labels
  labeled_run = int(np.floor(num_charts/2))

  ax[labeled_run].yaxis.set_label_text(f"Rainfall\n{test}")
  ax2 = ax[labeled_run].twinx()
  ax2.yaxis.set_label_text("Network-wide error (L2)")
  ax2.yaxis.set_ticklabels([])
  ax[num_charts-1].set_xlabel("time")

  plt.savefig(figname)
  # TODO: color using goldenrod `color=(218/255, 165/255, 32/255)` and black `color=(0, 0, 0)`

  plt.close()

  return test_subset


def plot_rain_vs_logloss(metadata, times, rain_data, L2, figname, test_subset=[]):
  # initialize charts
  num_test_runs = len(metadata['test_runs'])
  num_charts = max(min(6, num_test_runs), 2) # at least 2 charts to avoid subscripting issue
  fig, ax = plt.subplots(num_charts, 1, sharex=True)
  fig.set_size_inches((10,9))

  # annotate plot
  ax[0].title.set_text(f"Model trained on {metadata['dataset_size']} data points, validated on a hold-out 48-hr period.")

  # plot each test run's error
  time_index_dict = subdivide_time_series(times)
  test_indices = {}

  # choose at most 6 examples to plot
  if len(test_subset)==0:
    test_subset = np.random.permutation(
                    np.arange(num_test_runs)
                  )[:min(6,num_test_runs)]
  for ind, test in enumerate(test_subset):
    print(f"ind: {ind}, test: {test}")
    # parse time_index_dict
    test_indices[test] = []
    for _ind in range(len(time_index_dict)):
      test_indices[test].append(time_index_dict[_ind][test])
    # collect data from concatenated test data
    time_subseries = times.iloc[test_indices[test]] # t2.iloc[test_indices[test]]
    rain_subseries = rain_data.iloc[test_indices[test]]
    L2_subseries = L2.iloc[test_indices[test]].apply(np.log)
    # plot
    ax[ind].plot(time_subseries, rain_subseries)
    ax2 = ax[ind].twinx()
    ax2.plot(time_subseries, L2_subseries, color="red")
    ax[ind].yaxis.set_label_text(test)

  # make y-labels
  labeled_run = int(np.floor(num_charts/2))

  ax[labeled_run].yaxis.set_label_text(f"Rainfall\n{test}")
  ax2 = ax[labeled_run].twinx()
  ax2.yaxis.set_label_text("Network-wide error (log-L2)")
  ax2.yaxis.set_ticklabels([])
  ax[num_charts-1].set_xlabel("time")

  plt.savefig(figname)
  # TODO: color using goldenrod `color=(218/255, 165/255, 32/255)` and black `color=(0, 0, 0)`

  plt.close()

  return test_subset


def plot_samples(times, target, prediction, num_elements=4, num_series=1, figname="temp_sample_plots.png"):
  # prepare figure
  num_charts = min(num_elements, target.shape[1])
  fig, ax = plt.subplots(num_charts, num_series, sharex=True, sharey=True)
  if num_charts == 1: ax = [ax]
  if num_series == 1: ax = np.array([ax]).T
  fig.set_size_inches((12,7))
  # ax[0, 0].title(f"Sample of {num_charts} predictions vs. actuals on test data.")
  print(ax)

  def map_to_unity(series):
    # return (series - series.min()) / (series.max() - series.min())
    return series

  # select random elements to plot
  columns2plot = np.random.permutation(np.arange(target.shape[1]))[:num_charts]

  # select random subset of time series to plot
  try:
    series_length = times.nunique().to_numpy()[0] # creates array of length 1
  except:
    try:
      series_length = times.nunique()
    except:
      raise Error("Problem with `series_length` variable definition.")
  total_num_series = times.shape[0]//series_length
  series2plot = np.random.permutation(np.arange(total_num_series))[:num_series]

  # plot each element
  for (i, _c) in enumerate(columns2plot):
    for (j, _s) in enumerate(series2plot):
      # calculate time indices
      start_ind = _s * series_length
      end_ind = start_ind + series_length
      print(f"start_ind: {start_ind}, end_ind: {end_ind}")

      # plot
      ax[i][j].plot(
        times.apply(pd.to_datetime)[start_ind:end_ind].to_numpy().flatten(),
        map_to_unity(target.iloc[start_ind:end_ind, _c] 
          if isinstance(target, pd.DataFrame) 
          else target[start_ind:end_ind, 0, _c]),
        color='black',
        alpha=0.7
      )
      ax[i][j].plot(
        times.apply(pd.to_datetime)[start_ind:end_ind].to_numpy().flatten(),
        map_to_unity(prediction.iloc[start_ind:end_ind, _c]
          if isinstance(prediction, pd.DataFrame) 
          else prediction[start_ind:end_ind, 0, _c]),
        color='green',
        alpha=0.7
      )
      try:
        ax[i][j].set_xticks([pd.to_datetime(times.iloc[5,0]), pd.to_datetime(times.iloc[-5,0])])
      except:
        ax[i][j].set_xticks([pd.to_datetime(times.iloc[5]), pd.to_datetime(times.iloc[-5])])
    ax[i][0].set_ylabel(_c)

  plt.savefig(figname)
  plt.show()
  # plt.close()


def plot_worst_elmts(target, prediction, figname):
  err = (prediction - target).abs()

  fig, ax = plt.subplots(2,1)
  fig.set_size_inches((10,6))
  plt.title("Worst performing node & link on test data.")

  node_colms = list(filter(lambda x: x.find("invert_depth")>=0, target.columns))
  link_colms = list(filter(lambda x: x.find("capacity")>=0, target.columns))
  
  if len(node_colms) > 0:
    # find worst node
    node_err = err.get(node_colms)
    peggiore_nodo_ind = np.argmax(node_err.max())
    pegg_node = node_colms[peggiore_nodo_ind]
    
    # plot worst examples of prediction
    ax[0].plot(np.arange(target.shape[0]), target[pegg_node])
    ylims = ax[0].get_ylim()
    ax[0].plot(np.arange(target.shape[0]), prediction.get(pegg_node)) # [:, peggiore_nodo_ind])
    ax[0].set_ylim(ylims)
    ax[0].legend(["Actual", "Predicted"])

  if len(link_colms) > 0:
    # find worst link
    link_err = err.get(link_colms)
    peggiore_nesso_ind = np.argmax(link_err.mean())
    pegg_nesso = link_colms[peggiore_nesso_ind]

    # plot worst examples of prediction
    ax[1].plot(np.arange(target.shape[0]), target[pegg_nesso])
    ylims = ax[1].get_ylim()
    ax[1].plot(np.arange(target.shape[0]), prediction.get(pegg_nesso)) #[:, peggiore_nesso_ind])
    ax[1].set_ylim(ylims)
    ax[1].legend(["Actual", "Predicted"])

  plt.savefig(figname)
  # plt.show()
  plt.close()

  # plt.scatter(target[pegg_node], prediction[:, peggiore_nodo_ind])
  # plt.title("Simulation vs. Prediction for\nWorst Performing Node")
  # plt.show()

  # plt.scatter(target[pegg_nesso], prediction[:, peggiore_nesso_ind])
  # plt.title("Simulation vs. Prediction for\nWorst Performing Link")
  # plt.show()


def plot_best_elmts(target, prediction, figname):
  err = (prediction - target).abs()

  node_colms = list(filter(lambda x: x.find("invert_depth")>=0, target.columns))
  link_colms = list(filter(lambda x: x.find("capacity")>=0, target.columns))

  fig, ax = plt.subplots(2,1)
  fig.set_size_inches((10,6))
  plt.title("Best performing node & link on test data.")
  
  if len(node_colms) > 0:
    # find worst node
    node_err = err.get(node_colms).apply(abs)
    maggiore_nodo_ind = np.argmin(node_err.mean())
    magg_node = node_colms[maggiore_nodo_ind]
    
    # plot worst examples of prediction
    ax[0].plot(np.arange(target.shape[0]), target[magg_node])
    ylims = ax[0].get_ylim()
    ax[0].plot(np.arange(target.shape[0]), prediction.get(magg_node)) # [:, maggiore_nodo_ind])
    ax[0].set_ylim(ylims)
    ax[0].legend(["Actual", "Predicted"])
  if len(link_colms) > 0:
    # find worst link
    link_err = err.get(link_colms)
    maggiore_nesso_ind = np.argmin(link_err.mean())
    magg_nesso = link_colms[maggiore_nesso_ind]

    # plot worst examples of prediction
    ax[1].plot(np.arange(target.shape[0]), target[magg_nesso])
    ylims = ax[1].get_ylim()
    ax[1].plot(np.arange(target.shape[0]), prediction.get(magg_nesso)) # [:, maggiore_nesso_ind])
    ax[1].set_ylim(ylims)
    ax[1].legend(["Actual", "Predicted"])

  plt.savefig(figname)
  # plt.show()
  plt.close()


#





