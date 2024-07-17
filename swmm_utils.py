# utils.py
# Mathew Titus
# April, 2024
# 
# Support functions for handling pyswmm outputs.
# 
# TODO: May improve performance to use a single Output call.
# 
######################################################################

import pandas as pd
import numpy as np
from pyswmm import Output, LinkSeries, NodeSeries, SubcatchSeries, SystemSeries

def framify(dic, col):
  '''
  Turn a dictionary of time:value entries into a dataframe with given columns
  '''
  frame = pd.DataFrame(data=zip(dic.keys(), dic.values()), columns=col)
  return frame


def jsonify(df_dict):
  '''
  Replace the DataFrames of a dictionary with their JSON content
  '''
  return dict(map(
    lambda x: (x[0], x[1].to_json()),
    df_dict.items()
  ))


def link_data(output_path):
  '''
  Returns a dictionary indexed by link name, with dataframe values
  of all link time series from the sim:
    time, capacity, flow_depth, flow_rate, flow_velocity, flow_volume, pollut_conc_0
  '''
  data = {}
  print(f"Opening {output_path}")
  out = Output(output_path)
  print("Output loaded: {}".format(out))
  # with  as out:
  # list link objects
  link_names = list(out.links.keys())
  print(f"Link names: {link_names}")
  num_links = len(link_names)

  # get time series variables
  first_link_name = out.links.keys().__iter__().__next__()
  print(f"First link name: {first_link_name}")
  all_vars = list(LinkSeries(out)[first_link_name].__annotations__.keys())
  print("Variable list:")
  for var in all_vars: print(var)

  for _ in range(num_links):
    dfs = []

    name = link_names[_]
    print(f"Loading link {name}")
    link = LinkSeries(out)[name]
    print(f"Link loaded: {link}")

    # collect time series
    for var in all_vars:
      time_series = link.__getattr__(var)
      ts = framify(time_series, ["time", var])
      ts = ts.set_index("time")
      dfs.append(ts)

    # compile data
    df = pd.concat(dfs, axis=1)
    data[name] = df.reset_index()

  # close 
  out.close()

  # return dictionary of dataframes
  return data


def node_data(output_path):
  '''
  Returns a dictionary indexed by node name, with dataframe values
  of all node time series from the sim:
    time, capacity, flow_depth, flow_rate, flow_velocity, flow_volume, pollut_conc_0
  '''
  data = {}

  with Output(output_path) as out:
    # list link objects
    node_names = list(out.nodes.keys())
    num_nodes = len(node_names)

    # get time series variables
    first_node_name = out.nodes.keys().__iter__().__next__
    all_vars = list(NodeSeries(out)[first_node_name].__annotations__.keys())

    for _ in range(num_nodes):
      dfs = []

      name = node_names[_]
      node = NodeSeries(out)[name]

      # collect time series
      for var in all_vars:
        time_series = node.__getattr__(var)
        ts = framify(time_series, ["time", var])
        ts = ts.set_index("time")
        dfs.append(ts)

      # compile data
      df = pd.concat(dfs, axis=1)
      data[name] = df.reset_index()

    # return dictionary of dataframes
    return data


def raingage_data(output_path):
  '''
  Returns a dictionary indexed by raingage name, with dataframe values
  of all rain gage time series from the sim:
    time, capacity, flow_depth, flow_rate, flow_velocity, flow_volume, pollut_conc_0
  '''
  data = {}

  with Output(output_path) as out:
    # list link objects
    rg_names = list(out.raingages.keys())
    num_rgs = len(rg_names)

    # get time series variables
    first_rg_name = out.raingages.keys().__iter__().__next__
    all_vars = list(LinkSeries(out)[first_link_name].__annotations__.keys())

    for _ in range(num_links):
      dfs = []

      name = link_names[_]
      link = LinkSeries(out)[name]

      # collect time series
      for var in all_vars:
        time_series = link.__getattr__(var)
        ts = framify(time_series, ["time", var])
        ts = ts.set_index("time")
        dfs.append(ts)

      # compile data
      df = pd.concat(dfs, axis=1)
      data[name] = df.reset_index()

    # return dictionary of dataframes
    return data


def subcatch_data(output_path):
  '''
  Returns a dictionary indexed by subcatchment name, with dataframe values
  of all subcatchment time series from the sim:
    time, capacity, flow_depth, flow_rate, flow_velocity, flow_volume, pollut_conc_0
  '''
  data = {}

  with Output(output_path) as out:
    # list link objects
    subc_names = list(out.subcatchments.keys())
    num_subcs = len(subc_names)

    # get time series variables
    first_subc_name = out.subcatchments.keys().__iter__().__next__
    all_vars = list(SubcatchSeries(out)[first_subc_name].__annotations__.keys())

    for _ in range(num_subcs):
      dfs = []

      name = subc_names[_]
      subc = SubcatchSeries(out)[name]

      # collect time series
      for var in all_vars:
        time_series = subc.__getattr__(var)
        ts = framify(time_series, ["time", var])
        ts = ts.set_index("time")
        dfs.append(ts)

      # compile data
      df = pd.concat(dfs, axis=1)
      data[name] = df.reset_index()

    # return dictionary of dataframes
    return data


def get_time_series(output_path):
  '''
  Returns a dictionary indexed by time series name, with dataframe values
  of all subcatchment time series from the sim.
  '''
  data = {}

  with Output(output_path) as out:
    # list time series objects
    ts_names = list(SystemSeries(out).__annotations__.keys())
    num_tss = len(ts_names)

    # get time series variables
    # first_ts_name = out.subcs.keys().__iter__().__next__
    # all_vars = list(SystemSeries(out)[first_subc_name].__annotations__.keys())

    dfs = []
    for name in ts_names:
      # collect time series
      series = SystemSeries(out).__getattr__(name)
      ts = framify(series, ["time", name])
      ts = ts.set_index("time")
      dfs.append(ts)

    # compile data, close .out file
    data = pd.concat(dfs, axis=1)\
      .reset_index()

  # return dictionary of dataframes
  return data


def get_data(output_path):
  '''
  Collect the link, node, and subcatchment data from an output file.
  '''
  print(f"get_data call\n\toutput_path: {output_path}")
  data = {
    'links': link_data(output_path),
    'nodes': node_data(output_path),
    'series': get_time_series(output_path),
    'subcatchments': subcatch_data(output_path)
  }
  return data


def data_to_json(data):
  '''
  Convert data dictionary (dict of dicts of dataframes) to JSON & return.
  '''
  return dict(map(
      lambda x: (x[0], jsonify(x[1])),
      data.items()
    ))


def get_json(output_path):
  '''
  Formulate JSON description of output, return.
  '''
  return data_to_json(get_data(output_path))


def swmm_TS_template(name, days):
  '''

  '''

  output_string = ''
  for ind in range(days):
    output_string += f"{name}\t{ind}\t{np.round(np.random.rand(), 4)}\n"

  return output_string




import matplotlib.pyplot as plt

# TODO: 
def view_rainfall(output_path):
  '''
  '''
  # get list of raingages & a sample subcatchment for each gage

  # get the rainfall time series for each sample subc
  series = subcatch_data(output_path)

  # get total rainfall
  rain = get_time_series(output_path).get(['time', 'rainfall'])

  # plot each subcatchment in vertical subplots
  plt.plot(rain.time, rain.rainfall)
  plt.show()





# 


