import networkx as nx
import pandas as pd

##### Measurement Functions #####
def meas_graph_attr(G):
	meas=dict(G.graph)
	return meas

def meas_node_attr(G):
	meas = dict()
	attr = nx.get_node_attributes(G,G.nodes()[0])
	for a,val in attr.items():
		attr = nx.get_node_attributes(G,a)
		meas.update({(n,a):val for n in G.nodes()})
	
	return meas

def meas_edge_attr(G):
	meas = dict()
	nx.get_edge_attributes(G,G.edges()[0])
	for a,val in attr.items():
		attr = nx.get_node_attributes(G,a)
		meas.update({(e[0],e[1],a):val for e in G.edges()})

	return meas

##### Change Detection Functions #####
def get_value_changes(ds):
	''' Takes a data series and outputs (start,val) pairs - 
	one for each change in the value of the data series.
	'''
	changes = [(ds.index[0],ds[ds.index[0]])]

	return changes


##### File Output Functions #####

def build_xgmml_file(f,gdf,ndf,edf):
	''' This function builds the xml file given the file object f,
    a graph df, node df, and edge df. First it will look at when 
	attributes change, and then use that to decide when to add an 
	attribute tag.
	'''
	f.write(header_str)
	f.write(graph_start_str)
	t0 = gdf.index[0]
	tf = gdf.index[-1]

	i = 0
	for n in ndf.columns.get_level_values(0):
		values = {'label':str(n),'id':str(i),'start':t0,'end':t0}
		f.write(node_start.format(**values))
		
		for a in ndf.columns.get_level_values(1):
			changes = get_value_changes(ndf.loc[:,[n,a]])
			for c in range(len(changes[:-1])):
				values = {'name':a,'type':'real','value':changes[c][1],'start':changes[c][0],'end':changes[c+1][0]}
				f.write(node_start_str.format(**values))
		i += 1
		f.write(node_end_str)

	for u in edf.columns.get_level_values(0):
		for v in edf.columns.get_level_values(1):
			values = {'label':'(%s,%s)'%(str(u),str(v)),'source':str(u),'target':str(v),'start':ts[0],'end':ts[-1]}
			f.write(edge_start_str.format(**values))
			
			for a in ndf.columns.get_level_values(2):
				changes = get_value_changes(ndf.loc[:,(u,v,a)])
				for c in range(len(changes[:-1])):
					values = {'name':a,'type':'real','value':changes[c][1],'start':changes[c][0],'end':changes[c+1][0]}
					f.write(attr_str.format(**values))
			f.write(edge_end_str)

	f.write(graph_end_str)

	return


##### File Output Strings #####
header_str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n
<!-- Created using the networkxtimeseries library for python. -->\r\n'''

graph_start_str = '<graph label="testDD5" directed="0">\r\n'
graph_end_str = '</graph>'

node_start_str = '\t<node label="{label}" id="{id}" start="{start}" end="{end}">\r\n'
node_end_str = '\t</node>'

edge_start_str = '\t<edge label="edge_4_6" source="4" target="6" start="0" end="79">\r\n'
edge_end_str = '</edge>'

attr_str = '\t\t<att name="{name}" type="{type}" value="{value}" start="{start}" end="{end}"/>\r\n'
