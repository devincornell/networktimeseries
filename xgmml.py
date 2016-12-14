import networkx as nx
import pandas as pd

##### Change Detection Functions #####
def get_value_changes(ds):
	''' Takes a data series and outputs (start,val) pairs - 
	one for each change in the value of the data series.
	'''
	changes = [(ds.index[0],ds[ds.index[0]])]

	return changes


##### File Output Functions #####

def build_xgmml_file(f,ndf,edf):
	''' This function builds the xml file given the file object f,
    a graph df, node df, and edge df. First it will look at when 
	attributes change, and then use that to decide when to add an 
	attribute tag.
	'''
	f.write(header_str)
	f.write(graph_start_str)
	t0 = edf.index[0]
	tf = edf.index[-1]

	i = 0
	for n in ndf.columns.get_level_values(0):
		values = {'label':str(n),'id':str(i),'start':t0,'end':t0}
		f.write(node_start_str.format(**values))
		
		for a in ndf.columns.get_level_values(1):
			changes = get_value_changes(ndf.loc[:,(n,a)])
			for c in range(len(changes[:-1])):
				values = {'name':a,'type':'real','value':changes[c][1],'start':changes[c][0],'end':changes[c+1][0]}
				f.write(node_start_str.format(**values))
		i += 1
		f.write(node_end_str)

	for u in edf.columns.get_level_values(0):
		for v in edf.columns.get_level_values(1):
			values = {'label':'(%s,%s)'%(str(u),str(v)),'source':str(u),'target':str(v),'start':t0,'end':tf}
			f.write(edge_start_str.format(**values))
			
			for a in edf.columns.get_level_values(2):
				try:
					changes = get_value_changes(edf.loc[:,(u,v,a)])
				except:
					changes = get_value_changes(edf.loc[:,(v,u,a)])
				for c in range(len(changes[:-1])):
					values = {'name':a,'type':'real','value':changes[c][1],'start':changes[c][0],'end':changes[c+1][0]}
					f.write(attr_str.format(**values))
			f.write(edge_end_str)

	f.write(graph_end_str)

	return


##### File Output Strings #####
header_str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n
<!-- Created using the networkxtimeseries library for python. -->\n'''

graph_start_str = '<graph label="testDD5" directed="0">\n'
graph_end_str = '</graph>\n'

node_start_str = '\t<node label="{label}" id="{id}" start="{start}" end="{end}">\n'
node_end_str = '\t</node>\n'

edge_start_str = '\t<edge label="edge_4_6" source="4" target="6" start="0" end="79">\n'
edge_end_str = '\t</edge>\n'

attr_str = '\t\t<att name="{name}" type="{type}" value="{value}" start="{start}" end="{end}"/>\n'
