# swmm_utils.py
# Mathew Titus
# April, 2024
# 
# Support functions for handling pyswmm outputs.
# 
######################################################################

import pandas as pd
import numpy as np
from pyswmm import Output, LinkSeries, NodeSeries, SubcatchSeries, SystemSeries


# Temp sensor ids
temp_nodes = ["33-486014", "36-474088", "33-478019"]
temp_links = ["33-486014.1", "36-474088.1", "33-478019.1"]

# Permanent sensors
perm_nodes = ["33-486037", "36-486015", "36-486019", "33-488083", "36-486016"]
perm_links = ["33-486037.1", "36-486015.1", "36-486019.1", "33-488083.1", "36-486016.1"]

# # Eyeballed perm sensors
# wrong_perm_nodes = ["36-484057", "36-486016", "33-488083", "36-486002", "36-486023"]
# wrong_perm_links = ["36-484057.1", "36-486016.1", "33-488083.1", "36-486002.1", "36-486011.1"]

# Meso-network ids
meso_nodes = ["33-486014", "36-486019", "36-486023", "36-484035", "33-478019", "36-482015", "36-482055", 
              "33-472048", "30-474069", "36-484057", "39-480015", "39-476059", "39-476057", "39-478058", 
              "36-476026"]
meso_links = ["33-486014.1", "36-484035.1", "36-482015.1", "36-482055.1", "33-472048.1", "30-474069.1", 
              "39-480015.1", "39-476059.1", "39-476057.1", "39-478058.1", "36-476026.1"]


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


def link_data(out):
  '''
  Returns a dictionary indexed by link name, with dataframe values
  of all link time series from the sim:
    time, capacity, flow_depth, flow_rate, flow_velocity, flow_volume, pollut_conc_0
  '''
  data = {}
  # print(f"Opening {output_path}")
  # out = Output(output_path)
  # print("Output loaded: {}".format(out))

  # with  as out:
  # list link objects
  link_names = list(out.links.keys())
  # print(f"Link names: {link_names}")
  num_links = len(link_names)

  # get time series variables
  first_link_name = out.links.keys().__iter__().__next__()
  # print(f"First link name: {first_link_name}")
  all_vars = list(LinkSeries(out)[first_link_name].__annotations__.keys())
  print("Variable list:")
  for var in all_vars: print(var)

  for _ in range(num_links):
    dfs = []

    name = link_names[_]
    # print(f"Loading link {name}")
    link = LinkSeries(out)[name]
    # print(f"Link loaded: {link}")

    # collect time series
    for var in all_vars:
      time_series = link.__getattr__(var)
      ts = framify(time_series, ["time", var])
      ts = ts.set_index("time")
      dfs.append(ts)

    # compile data
    df = pd.concat(dfs, axis=1)
    data[name] = df.reset_index()

  # # close 
  # out.close()

  # return dictionary of dataframes
  return data


def node_data(out):
  '''
  Returns a dictionary indexed by node name, with dataframe values
  of all node time series from the sim:
    invert_depth, hydraulic_head, ponded_volume, lateral_inflow, total_inflow, flooding_losses, pollut_conc_0
  '''
  data = {}

  # with Output(output_path) as out:
    # list link objects
  node_names = list(out.nodes.keys())
  num_nodes = len(node_names)

  # get time series variables
  first_node_name = out.nodes.keys().__iter__().__next__
  all_vars = list(NodeSeries(out)[first_node_name].__annotations__.keys())

  print("Variable list:")
  for var in all_vars: print(var)

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


def raingage_data(out):
  '''
  Returns a dictionary indexed by raingage name, with dataframe values
  of all rain gage time series from the sim:
    time, capacity, flow_depth, flow_rate, flow_velocity, flow_volume, pollut_conc_0
  '''
  data = {}

  # with Output(output_path) as out:
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


def subcatch_data(out):
  '''
  Returns a dictionary indexed by subcatchment name, with dataframe values
  of all subcatchment time series from the sim:
    time, capacity, flow_depth, flow_rate, flow_velocity, flow_volume, pollut_conc_0
  '''
  data = {}

  # with Output(output_path) as out:
  # list link objects
  subc_names = list(out.subcatchments.keys())
  num_subcs = len(subc_names)

  # get time series variables
  first_subc_name = out.subcatchments.keys().__iter__().__next__
  all_vars = list(SubcatchSeries(out)[first_subc_name].__annotations__.keys())

  print("Variable list:")
  for var in all_vars: print(var)

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


def get_time_series(out):
  '''
  Returns a dictionary indexed by time series name, with dataframe values
  of all subcatchment time series from the sim.
  '''
  data = {}

  # with Output(output_path) as out:
  # list time series objects
  ts_names = list(SystemSeries(out).__annotations__.keys())
  num_tss = len(ts_names)

  # get time series variables
  # first_ts_name = out.subcs.keys().__iter__().__next__
  # all_vars = list(SystemSeries(out)[first_subc_name].__annotations__.keys())

  print("Subcatchment `ts_names`:")
  print(ts_names)

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

  # open
  print(f"Opening {output_path}")
  out = Output(output_path)
  print("Output loaded: {}".format(out))

  # extract
  data = {
    'links': link_data(out),
    'nodes': node_data(out),
    'series': get_time_series(out),
    'subcatchments': subcatch_data(out)
  }

  # close 
  out.close()

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


def plot_monitor_data(data):
  import matplotlib.pyplot as plt
  
  # filtered meso_links of temp_ or perm_links content
  node_data = dict.fromkeys(temp_nodes + perm_nodes + meso_nodes)
  [node_data.update({key: data['nodes'][key]}) for key in node_data.keys()]
  link_data = dict.fromkeys(temp_links + perm_links + meso_links)
  [link_data.update({key: data['links'][key]}) for key in link_data.keys()]

  # node fig setup
  fig, ax = plt.subplots(4,1, figsize=(12,10))
  # plot rain
  ax[3].plot(data['series'].time, data['series'].rainfall)
  # plot invert_depths
  for _ in temp_nodes:
    ax[0].plot(node_data[_].time, node_data[_].invert_depth)
  ax[0].legend(list(node_data.keys()))
  for _ in perm_nodes:
    ax[1].plot(node_data[_].time, node_data[_].invert_depth)
  ax[1].legend(list(node_data.keys()))
  for _ in meso_nodes:
    ax[2].plot(node_data[_].time, node_data[_].invert_depth)
  ax[2].legend(list(node_data.keys()))
  # annotate & save
  # plt.title("Invert Depth")
  ax[3].title.set_text('Invert Depth')
  plt.savefig("latest_run_node_data.png")
  plt.close()

  # node fig setup
  fig, ax = plt.subplots(4,1, figsize=(12,10))
  # plot rain
  ax[3].plot(data['series'].time, data['series'].rainfall)
  # plot invert_depths
  for _ in temp_links:
    ax[0].plot(link_data[_].time, link_data[_].capacity)
  ax[0].legend(list(link_data.keys()))
  for _ in perm_links:
    ax[1].plot(link_data[_].time, link_data[_].capacity)
  ax[1].legend(list(link_data.keys()))
  for _ in meso_links:
    ax[2].plot(link_data[_].time, link_data[_].capacity)
  ax[2].legend(list(link_data.keys()))
  # annotate & save
  # fig.title("Capacity")
  plt.savefig("latest_run_link_data.png")
  plt.close()

  return node_data, link_data



# import matplotlib.pyplot as plt

# # TODO: 
# def view_rainfall(output_path):
#   '''
#   '''
#   # get list of raingages & a sample subcatchment for each gage

#   # get the rainfall time series for each sample subc
#   series = subcatch_data(output_path)

#   # get total rainfall
#   rain = get_time_series(output_path).get(['time', 'rainfall'])

#   # plot each subcatchment in vertical subplots
#   plt.plot(rain.time, rain.rainfall)
#   plt.show()





# 


