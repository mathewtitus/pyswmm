# learning_swmmio.py

import swmmio

l2w_path = "/mnt/c/Users/litso/Documents/EPA SWMM Projects/"

#instantiate a swmmio model object
mymodel = swmmio.Model(f'{l2w_path}')

# Pandas dataframe with most useful data related to model nodes, conduits, and subcatchments
nodes = mymodel.nodes.dataframe
links = mymodel.links.dataframe
subs = mymodel.subcatchments.dataframe

#enjoy all the Pandas functions
nodes.head()

#write to a csv
nodes.to_csv(f'{l2w_path}nodes.csv')

#calculate average and weighted average impervious
avg_imperviousness = subs.PercImperv.mean()
weighted_avg_imp = (subs.Area * subs.PercImperv).sum() / len(subs)



# pass custom init arguments into the Nodes object instead of using default settings referenced by mymodel.nodes() 
nodes = swmmio.Nodes(
    model=mymodel, 
    inp_sections=['junctions', 'storages', 'outfalls'],
    rpt_sections=['Node Depth Summary', 'Node Inflow Summary'],
    columns=[ 'InvertElev', 'MaxDepth', 'InitDepth', 'SurchargeDepth', 'MaxTotalInflow', 'coords']
)

# access data 
nodes.dataframe 

# save images
img = swmmio.draw_model(mymodel)
img.save('file/path/to/save')



# Plotting a few things

#isolate nodes that have flooded for more than 30 minutes
flooded_series = nodes.loc[nodes.HoursFlooded>0.5, 'TotalFloodVol']
flood_vol = sum(flooded_series) #total flood volume (million gallons)
flooded_count = len(flooded_series) #count of flooded nodes

#highlight these nodes in a graphic
nodes['draw_color'] = '#787882' #grey, default node color
nodes.loc[nodes.HoursFlooded>0.5, 'draw_color'] = '#751167' #purple, flooded nodes

#set the radius of flooded nodes as a function of HoursFlooded
nodes.loc[nodes.HoursFlooded>1, 'draw_size'] = nodes.loc[nodes.HoursFlooded>1, 'HoursFlooded'] * 12

# pass custom init arguments into the Nodes object instead of using default settings referenced by m.nodes() 
conds = Condu(
    model=m, 
    inp_sections=['junctions', 'storages', 'outfalls'],
    rpt_sections=['Node Depth Summary', 'Node Inflow Summary'],
    columns=[ 'InvertElev', 'MaxDepth', 'InitDepth', 'SurchargeDepth', 'MaxTotalInflow', 'coords']
)

#make the conduits grey, sized as function of their geometry
conds['draw_color'] = '#787882'
conds['draw_size'] = conds.Geom1

#add an informative annotation, and draw:
annotation = 'Flooded Volume: {}MG\nFlooded Nodes:{}'.format(round(flood_vol), flooded_count)
swmmio.draw_model(mymodel, annotation=annotation, file_path='flooded_anno_example.png')


#add an informative annotation, and draw:
annotation = 'Flooded Volume: {}MG\nFlooded Nodes:{}'.format(round(flood_vol), flooded_count)
swmmio.draw_model(mymodel, annotation=annotation, file_path='flooded_anno_example.png')

