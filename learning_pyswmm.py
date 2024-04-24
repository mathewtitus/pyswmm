# learning_pyswmm.py

import pyswmm

from pyswmm import Simulation, Nodes, Links

with Simulation(r'Example1.inp') as sim:
    Node21 = Nodes(sim)["21"]
    print("Invert Elevation: {}". format(Node21.invert_elevation))

    Link15 = Links(sim)['15']
    print("Outlet Node ID: {}".format(Link15.outlet_node))

    # Launch a simulation!
    for ind, step in enumerate(sim):
        if ind % 100 == 0:
            print(sim.current_time,",",round(sim.percent_complete*100),"%",\
                  Node21.depth, Link15.flow)


from pyswmm import Output, SubcatchSeries, NodeSeries, LinkSeries, SystemSeries

with Output('model.out') as out:
    print(len(out.subcatchments))
    print(len(out.nodes))
    print(len(out.links))
    print(out.version)

    sub_ts = SubcatchSeries(out)['S1'].runoff_rate
    node_ts = NodeSeries(out)['J1'].invert_depth
    link_ts = LinkSeries(out)['C2'].flow_rate
    sys_ts = SystemSeries(out).rainfall
