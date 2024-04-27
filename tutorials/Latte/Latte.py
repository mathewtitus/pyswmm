'''
PySWMM Latte Code
Author: Bryant McDonnell
Version: 2
Date: Jan 10, 2023
'''
from pyswmm import Simulation, Nodes, Links

with Simulation(r'Example1.inp') as sim:
    ########################
    # Simulation information
    # remaining references are available here:
    # https://pyswmm.readthedocs.io/en/stable/reference/simulation.html#pyswmm.simulation.Simulation
    print("Simulation info")
    flow_units = sim.flow_units
    print("Flow Units: {}".format(flow_units))
    system_units = sim.system_units
    print("System Units: {}".format(system_units))
    print("Start Time: {}".format(sim.start_time))
    print("Start Time: {}".format(sim.end_time))

    ##################
    # Node Information
    # remaining references are available here:
    # https://pyswmm.readthedocs.io/en/stable/reference/nodes.html#pyswmm.nodes.Node
    Node21 = Nodes(sim)["21"]
    print("Node 21 info")
    print("Invert Elevation: {}".format(Node21.invert_elevation))
    print("Physical Depth: {}".format(Node21.full_depth))
    print("Is it a Junction?: {}".format(Node21.is_junction()))

    ##################
    # Link Information
    # remaining references are available here:
    # https://pyswmm.readthedocs.io/en/stable/reference/links.html#pyswmm.links.Link
    Link15 = Links(sim)['15']
    print("Link 15 info")
    print("Inlet Node ID: {}".format(Link15.inlet_node))
    print("Outlet Node ID: {}".format(Link15.outlet_node))

    # Launch a simulation!
    for ind, step in enumerate(sim):
        if ind % 100 == 0:
            print(sim.current_time,",",round(sim.percent_complete*100),"%",\
                  Node21.depth, Link15.flow)

    node21_stat_out = Node21.statistics
    print(node21_stat_out)
    print("Max Node 15 Depth: {}".format(node21_stat_out['max_depth']))

    link15_stat_out = Link15.conduit_statistics
    print("Link 15 Peak Velocity: {}".format(link15_stat_out["peak_velocity"]))
