# REFERENCES

## swmm-api
https://pypi.org/project/swmm-api


## pyswmm
https://github.com/pyswmm


## swmmio
https://github.com/pyswmm/swmmio


## SWMM
https://github.com/pyswmm/Stormwater-Management-Model







# Using swmmio:
## Multithreading
If you have many models to run and would like to take advantage of your machine's cores, you can start a pool of simulations with the `--start_pool` (or `-sp`) command. After pointing `-sp` to one or more directories, swmmio will search for SWMM .inp files and add all them to a multiprocessing pool. By default, `-sp` leaves 4 of your machine's cores unused. This can be changed via the `-cores_left` argument.
```
$ #run all models in models in directories Model_Dir1 Model_Dir2
$ python -m swmmio -sp Model_Dir1 Model_Dir2  

$ #leave 1 core unused
$ python -m swmmio -sp Model_Dir1 Model_Dir2  -cores_left=1
```
## Modifying .inp files
Starting with a base SWMM model, other models can be created by inserting altered data into a new inp file. Useful for sensitivity analysis or varying boundary conditions, models can be created using a fairly simple loop, leveraging the `modify_model` package.

```
import os
import swmmio

#initialize a baseline model object
baseline = swmmio.Model(r'path\to\baseline.inp')
rise = 0.0 #set the starting sea level rise condition

#create models up to 5ft of sea level rise.
while rise <= 5:

    #create a dataframe of the model's outfalls
    outfalls = baseline.inp.outfalls

    #create the Pandas logic to access the StageOrTimeseries column of  FIXED outfalls
    slice_condition = outfalls.OutfallType == 'FIXED', 'StageOrTimeseries'

    #add the current rise to the outfalls' stage elevation
    outfalls.loc[slice_condition] = pd.to_numeric(outfalls.loc[slice_condition]) + rise
    baseline.inp.outfalls = outfalls
    
    #copy the base model into a new directory    
    newdir = os.path.join(baseline.inp.dir, str(rise))
    os.mkdir(newdir)
    newfilepath = os.path.join(newdir, baseline.inp.name + "_" + str(rise) + '_SLR.inp')
    
    #Overwrite the OUTFALLS section of the new model with the adjusted data
    baseline.inp.save(newfilepath)

    #increase sea level rise for the next loop
    rise += 0.25
```

### Deprecation (4/23)
Edited the `utils/text.py` and `utils/dataframes.py` swmmio files (.venv310/lib/python3.10/site-packages/swmmio/...) to replace `delimited_whitespace=True` with `sep='\s+'`.

