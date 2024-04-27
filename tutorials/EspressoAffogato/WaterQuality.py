'''
PySWMM and StormReactor Code Example A
Author: Brooke Mason
Version: 1
Date: December 21, 2022
'''
# Import libraries
from pyswmm import Simulation, Nodes
from StormReactor import waterQuality
import matplotlib.pyplot as plt

# Define water quality configuration dictionary
config = {'17': {'type': 'node', 'pollutant': 'TSS', 'method': 'EventMeanConc', 'parameters': {'C': 50.0}}}

# Create lists to save TSS results
UpstreamNode_TSS = []
WQNode_TSS = []
OutfallNode_TSS = []

# Initalize SWMM simulation
with Simulation(r'Example1_WQ.inp') as sim:
    # Node information
    UpstreamNode = Nodes(sim)['24']
    WQNode = Nodes(sim)['17']
    OutfallNode = Nodes(sim)['18']

    # Initialize StormReactor
    WQ = waterQuality(sim, config)

    # Launch a simulation
    for step in sim:
        # Update water quality each simulation step
        WQ.updateWQState()
        # Get and save TSS concentrations
        UpstreamNode_TSS.append(UpstreamNode.pollut_quality['TSS'])
        WQNode_TSS.append(WQNode.pollut_quality['TSS'])
        OutfallNode_TSS.append(OutfallNode.pollut_quality['TSS'])

# Plot TSS concentrations
plt.plot(UpstreamNode_TSS, ':', label="Upstream Node")
plt.plot(WQNode_TSS, '-', label="WQ Node")
plt.plot(OutfallNode_TSS, '--',label="Outfall")
plt.xlabel("Time")
plt.ylabel("TSS (mg/L)")
plt.legend()
plt.show()
