'''
PySWMM Espresso Machine
Author: Bryant McDonnell
Version: 1
Date: May 4, 2023
'''
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyswmm import Simulation, Nodes, Links, SystemStats


print("Running the Model without Control")
with Simulation(r'DemoModel.inp') as sim:
    system_stats = SystemStats(sim)

    # Interceptor nodes to observe
    J1 = Nodes(sim)["J1"]

    # Overflows to Observe
    OF1 = Nodes(sim)["OF1"]
    OF2 = Nodes(sim)["OF2"]
    OF3 = Nodes(sim)["OF3"]
    J8 = Nodes(sim)["J8"]

    # Initializing Data Arrays for timeseries plot
    ts_no_control=[]
    flooding_no_control=[]

    sim.step_advance(300)
    for ind, step in enumerate(sim):
        ts_no_control.append(sim.current_time)
        flooding_no_control.append(J1.flooding)
        pass
    # Get Results for Post Processing Table
    no_control_J8 = J8.cumulative_inflow
    no_control_total_overflow = OF1.cumulative_inflow \
                              + OF2.cumulative_inflow \
                              + OF3.cumulative_inflow
    no_controls = system_stats.routing_stats

print("\nRunning the Model with SIMPLE Control")
with Simulation(r'DemoModel.inp',
                'DemoModel_wControl.rpt',
                'DemoModel_wControl.out') as sim:
    system_stats = SystemStats(sim)

    # Instantiating the Orifices to Control
    OR1 = Links(sim)["OR1"]
    OR2 = Links(sim)["OR2"]
    OR3 = Links(sim)["OR3"]

    # Interceptor Nodes to Observe
    J1 = Nodes(sim)["J1"]
    J2 = Nodes(sim)["J2"]
    J8 = Nodes(sim)["J8"]

    # Storage Tanks to Observe
    SU1 = Nodes(sim)['SU1']
    SU2 = Nodes(sim)['SU2']
    SU3 = Nodes(sim)['SU3']

    # Overflows to Observe
    OF1 = Nodes(sim)["OF1"]
    OF2 = Nodes(sim)["OF2"]
    OF3 = Nodes(sim)["OF3"]

    # Initializing Data Arrays for timeseries plot
    ts_w_control=[]
    flooding_w_control=[]

    sim.step_advance(300)
    # Launch a simulation!
    in_wet_weather = False
    for ind, step in enumerate(sim):
        ts_w_control.append(sim.current_time)
        flooding_w_control.append(J1.flooding)
        if J2.depth > 4.5 and in_wet_weather == False:
            OR1.target_setting = 0.15
            OR2.target_setting = 0.15
            OR3.target_setting = 0.15
            in_wet_weather = True
        elif J2.depth <= 4 and in_wet_weather == True:
            OR1.target_setting = 0.25
            OR2.target_setting = 0.25
            OR3.target_setting = 0.25
        elif J2.depth < 2:
            OR1.target_setting = 1
            OR2.target_setting = 1
            OR3.target_setting = 1
            in_wet_weather = False

    # Performance Analysis (KPI Compare)
    w_control_J8 = J8.cumulative_inflow
    w_control_total_overflow = OF1.cumulative_inflow \
                             + OF2.cumulative_inflow \
                             + OF3.cumulative_inflow
    w_controls = system_stats.routing_stats


# BUILD TABLE TO COMPARE MODEL PERFORMANCE
u_convert=7.481/1.e6 # ft3->MG
print("|{:27s}|{:15s}|{:15s}|".format(" Volume Type  "," No Control (MG) ", " W Control (MG)  "))
print("| ------------------------- | --------------- | --------------- |")
for key in w_controls.keys():
    if key not in ["dry_weather_inflow", "evaporation_loss",\
                   "groundwater_inflow","II_inflow","reacted","seepage_loss",\
                   "wet_weather_inflow", "routing_error"]:
        print("| {:25s} | {:15.3f} | {:15.3f} |".format(key,
                                                        no_controls[key]*u_convert,
                                                        w_controls[key]*u_convert))
print("| {:25s} | {:15.3f} | {:15.3f} |".format("CSO Volume (OF1+OF2+OF3)",
                                                no_control_total_overflow*u_convert,
                                                w_control_total_overflow*u_convert))
print("| {:25s} | {:15.3f} | {:15.3f} |".format("Downstream Volume (J8)",
                                                no_control_J8*u_convert,
                                                w_control_J8*u_convert))





fig = plt.figure(figsize=(8,4), dpi=200) #Inches Width, Height
fig.suptitle("Model Flooding Compare at Node J1")
# Plot from the results compiled during simulation time
axis_1 = fig.add_subplot(1,1,1)
axis_1.plot(ts_no_control, flooding_no_control, '-b', label="No Control")
axis_1.plot(ts_w_control, flooding_w_control, ':g', label="With Control")
axis_1.set_ylabel("Flooding (MGD)")
axis_1.grid("xy")
axis_1.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("flooding.PNG")
plt.close()


# BONUS PROFILE PLOT!
#####################################################################
################### REPORT GENERATION ###############################
#####################################################################
try:
    swmmio_exists=True
    import swmmio
    from swmmio import find_network_trace
    from swmmio import (build_profile_plot, add_hgl_plot, add_node_labels_plot,
                        add_link_labels_plot)
except:
    swmmio_exists = False


if swmmio_exists:
    # Profile Plotter Demo
    rpt = swmmio.rpt("DemoModel.rpt")
    profile_depths_no_control = rpt.node_depth_summary.MaxNodeDepthReported
    rpt = swmmio.rpt("DemoModel_wControl.rpt")
    profile_depths_w_control = rpt.node_depth_summary.MaxNodeDepthReported

    mymodel = swmmio.Model(r"DemoModel.inp")
    fig = plt.figure(figsize=(11,9))
    fig.suptitle("Max HGL")
    ax = fig.add_subplot(6,1,(1,3))
    path_selection = find_network_trace(mymodel, 'J1', 'J8')
    profile_config = build_profile_plot(ax, mymodel, path_selection)
    add_hgl_plot(ax, profile_config, depth=profile_depths_no_control, label="No Control")
    add_hgl_plot(ax, profile_config, depth=profile_depths_w_control, color='green',label="With Control")
    add_node_labels_plot(ax, mymodel, profile_config)
    add_link_labels_plot(ax, mymodel, profile_config)
    leg = ax.legend()
    ax.grid('xy')
    ax.get_xaxis().set_ticklabels([])

    ax = fig.add_subplot(6,1,4)
    path_selection = find_network_trace(mymodel, 'J22', 'J1')
    profile_config = build_profile_plot(ax, mymodel, path_selection)
    add_hgl_plot(ax, profile_config, depth=profile_depths_no_control, label="No Control")
    add_hgl_plot(ax, profile_config, depth=profile_depths_w_control, color='green',label="With Control")
    add_node_labels_plot(ax, mymodel, profile_config)
    ax.grid('xy')
    ax.get_xaxis().set_ticklabels([])

    ax = fig.add_subplot(6,1,5)
    path_selection = find_network_trace(mymodel, 'J10', 'J3')
    profile_config = build_profile_plot(ax, mymodel, path_selection)
    add_hgl_plot(ax, profile_config, depth=profile_depths_no_control, label="No Control")
    add_hgl_plot(ax, profile_config, depth=profile_depths_w_control, color='green',label="With Control")
    add_node_labels_plot(ax, mymodel, profile_config)
    ax.grid('xy')
    ax.get_xaxis().set_ticklabels([])

    ax = fig.add_subplot(6,1,6)
    path_selection = find_network_trace(mymodel, 'J15', 'J6')
    profile_config = build_profile_plot(ax, mymodel, path_selection)
    add_hgl_plot(ax, profile_config, depth=profile_depths_no_control, label="No Control")
    add_hgl_plot(ax, profile_config, depth=profile_depths_w_control, color='green',label="With Control")
    add_node_labels_plot(ax, mymodel, profile_config)

    ax.grid('xy')
    fig.tight_layout()
    fig.savefig("profiles.png")
    plt.close()
