'''
PySWMM Cafe Con Leche
Author: Bryant McDonnell
Version: 1
Date: Jan 3, 2023
'''
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyswmm import Simulation, Nodes, Links, Output

with Simulation(r'Example2.inp') as sim:
    node_a = Nodes(sim)["16109"] # Injection Node
    node_b = Nodes(sim)["82309"] # Upstream Node
    node_c = Nodes(sim)["16009"] # Downstream Node

    # Initialize Lists for storing data
    time_stamps = []
    node_total_flow = []
    node_lateral_inflow = []
    us_head = []
    ds_head = []

    sim.step_advance(300)
    # Launch a simulation!
    for ind, step in enumerate(sim):
        if sim.current_time >= datetime.datetime(2002, 1, 1, 4, 0, 0):
            node_a.generated_inflow(20) # CFS into node
        time_stamps.append(sim.current_time)
        node_total_flow.append(node_a.total_inflow)
        node_lateral_inflow.append(node_a.lateral_inflow)
        us_head.append(node_b.head)
        ds_head.append(node_c.head)


# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure
fig = plt.figure(figsize=(8,4), dpi=200) #Inches Width, Height
fig.suptitle("Injection Node - 16109")
# Plot from the results compiled during simulation time
axis_1 = fig.add_subplot(2,1,1)
axis_1.plot(time_stamps, node_total_flow, '-g', label="Total Inflow")
axis_1.plot(time_stamps, node_lateral_inflow, ':b', label="Lateral Inflow")
axis_1.set_ylabel("Flow (CFS)")
#axis_1.get_xticklabels().set_visible(False) # turns off the labels
axis_1.grid("xy")
axis_1.legend()
# Second Axis
axis_2 = fig.add_subplot(2,1,2, sharex = axis_1)
axis_2.plot(time_stamps, us_head, '-g', label="Upstream - 82309")
axis_2.plot(time_stamps, ds_head, ':b', label="Downstream - 16009")
axis_2.set_ylabel("Head (ft)")
#axis_1.get_xticklabels().set_visible(False) # turns off the labels
axis_2.grid("xy")
axis_2.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("TEST.PNG")
plt.show()
