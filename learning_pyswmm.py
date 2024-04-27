# learning_pyswmm.py
# Mathew Titus, The Prediction Lab
# April, 2024
# 
# exec(open("/Users/mtitus/Documents/GitHub/COS_WW/pyswmm/learning_pyswmm.py").read())
# exec(open("learning_pyswmm.py").read())
# 
###########################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import swmm_utils as su
import swmm_timeseries as st
from pyswmm import Simulation, Nodes, Links, Output, SubcatchSeries, NodeSeries, LinkSeries, RainGages, SystemSeries, SimulationPreConfig, Subcatchments

from pandas import DataFrame as df
from pandas import concat

sim_path = r'./tutorials/Latte/Example1b.inp'
run_sim = False;

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


# output_filepath = '../tutorials/Latte/Example1b.out'
# out = Output(output_filepath)

# js = su.get_data(output_filepath)
# series = su.get_time_series(output_filepath)

# # show rain gage activity
# s1 = js['series']['rainfall']
# s2 = js['subcatchments']['7']['rainfall']
# s3 = js['subcatchments']['8']['rainfall']

# plt.plot(s1)
# # plt.plot(s2)
# # plt.plot(s3)



# # Create Config Handle
# sim_conf = SimulationPreConfig()

# # Specifying the update parameters
# # Parameter Order:
# # Section, Object ID, Parameter Index, New Value, Obj Row Num (optional)
# # sim_conf.add_update_by_token("SUBCATCHMENTS", "S1", 2, "J2")
# sim_conf.add_update_by_token("TIMESERIES", "TS1", 2, 8.0, 5)

# with Simulation(sim_path, outputfile="./tutorials/Latte/ex_1c.out", sim_preconfig = sim_conf) as sim:
#     # S1 = Subcatchments(sim)["S1"]
#     # print(S1.connection)

#     for step in sim:
#         pass

output_filepath = "./tutorials/Latte/ex_1c.out"
js = su.get_data(output_filepath)

times = st.get_times(output_filepath)
# data = np.abs(2 * np.random.randn(len(times)))
data = np.abs(2 * np.random.randn(11))

new_ts = pd.DataFrame([times, data])
new_ts = new_ts.T.iloc[:11]

sim_conf = SimulationPreConfig()
sim_conf = st.apply_time_series(sim_path, sim_conf, "TS2", new_ts.iloc[:,1])

with Simulation(sim_path, outputfile="./tutorials/Latte/new_ts_ex.out", sim_preconfig = sim_conf) as sim:
    for step in sim:
        pass








#



