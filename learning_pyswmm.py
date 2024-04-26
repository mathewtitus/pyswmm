# learning_pyswmm.py
# Mathew Titus, The Prediction Lab
# April, 2024
# 
# exec(open("learning_pyswmm.py").read())
# 
###########################################################################

import matplotlib.pyplot as plt
import pyswmm
from pyswmm import Simulation, Nodes, Links
from pyswmm import Output, SubcatchSeries, NodeSeries, LinkSeries, RainGages, SystemSeries

import swmm_utils as su

from pandas import DataFrame as df
from pandas import concat

sim_path = r'../tutorials/Latte/Example1b.inp'
run_sim = True;

if run_sim:
    # with Simulation(r'../tutorials/Latte/Example1.inp') as sim:
    sim = Simulation(sim_path)
    # Node21 = Nodes(sim)["21"]
    # print("Invert Elevation: {}". format(Node21.invert_elevation))
    # Link15 = Links(sim)['15']
    # print("Outlet Node ID: {}".format(Link15.outlet_node))
    # Launch a simulation!
    for ind, step in enumerate(sim):
        pass
        # if ind % 100 == 0:
            # print(sim.current_time,",",round(sim.percent_complete*100),"%\t",\
                  # round(Node21.depth, 5), "\t", round(Link15.flow, 5))
    rg = RainGages(sim)
    # input("")
    sim.close()


# output_filepath = '../tutorials/Latte/Example1.out'
# out = Output(output_filepath)
# js = su.get_data(output_filepath)
# s1 = js['series']['rainfall']
# s2 = js['subcatchments']['7']['rainfall']
# plt.plot(s1-0.001)
# plt.plot(s2)

output_filepath = '../tutorials/Latte/Example1b.out'
out = Output(output_filepath)
# with Output('../tutorials/Latte/Example1.out') as out:
    # print("Subcatchments: {}".format(len(out.subcatchments)))
# print("Nodes: {}".format(len(out.nodes)))
# print("Links: {}".format(len(out.links)))
# print("Version: {}".format(out.version))
# ls = SubcatchSeries(out)[1]

# sys_ts = SystemSeries(out).rainfall

# data = concat([runoff_rate, evap_loss, gw_outflow_rate, gw_table_elev], axis=1)
# print(data.head())

js = su.get_data(output_filepath)
# series = su.get_time_series(output_filepath)

# show rain gage activity
s1 = js['series']['rainfall']
s2 = js['subcatchments']['7']['rainfall']
s3 = js['subcatchments']['8']['rainfall']

plt.plot(s1)
# plt.plot(s2)
# plt.plot(s3)

from pyswmm import Simulation, SimulationPreConfig, Subcatchments

# Create Config Handle
sim_conf = SimulationPreConfig()

# Specifying the update parameters
# Parameter Order:
# Section, Object ID, Parameter Index, New Value, Obj Row Num (optional)
# sim_conf.add_update_by_token("SUBCATCHMENTS", "S1", 2, "J2")
sim_conf.add_update_by_token("TIMESERIES", "TS1", 2, 2, 5)

with Simulation(sim_path, sim_preconfig = sim_conf) as sim:
    # S1 = Subcatchments(sim)["S1"]
    # print(S1.connection)

    for step in sim:
        pass

js = su.get_data(output_filepath)

s1 = js['series']['rainfall']
plt.plot(s1+0.001)

# plt.legend(['rain_1a', 'rg1_a', 'rain_1b', 'rg1_b', 'rg2_b', 'rain_1c'])
plt.legend(['rain_1a', 'rain_1b', 'rain_1c'])
plt.show()


