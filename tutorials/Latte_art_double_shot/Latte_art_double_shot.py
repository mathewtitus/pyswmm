'''
PySWMM Latte Art Double Shot Code
Author: Bryant McDonnell
Version: 1
Date: Dec 8, 2022
'''

from pyswmm import Simulation
import swmmio

# Run Simulation PySWMM
with Simulation('./Example1.inp') as sim:
    for step in sim:
        pass

# Pass Give SWMM Simulation Artifacts to swmmio
crs = 'epsg:3728' # Coordinate Reference System
simulation_info = swmmio.Model("./Example1.inp", crs=crs)
swmmio.create_map(simulation_info, filename="test.html")
