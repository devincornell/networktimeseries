import networkx as nx
import pandas as pd

##### Measurement Functions #####
def meas_graph_attr(G):
	meas=dict(G.graph)
	return meas

def meas_node_attr(G):
	meas = dict()
	nx.get_node_attributes(G,G.nodes()[0])
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

	pass

##### File Output Functions #####
def build_xgmml_file(f,gdf,ndf,edf):
	''' This function builds the xml file given the file object f,
    a graph df, node df, and edge df. First it will look at when 
	attributes change, and then use that to decide when to add an 
	attribute tag.
	'''
	f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
	f.write('<!-- Created using the networkxtimeseries library for python. -->')
	f.write('<graph label="testDD5" directed="0">')

	for n in ndf.columns:
		f.write('\t<node label="node_1" id="1" start="0" end="100">')
		changes = get_value_changes(ndf[n])
		for c in changes:
			f.write('\t\t<att name="size" type="real" value="82" start="0" end="100"/>')
		f.write('\t</node>')

	for e in edf.columns:
		f.write('\t<edge label="edge_4_6" source="4" target="6" start="0" end="79">')
		changes = get_value_changes(ndf[e])
		for c in changes:
			f.write('\t\t<att name="weight" type="real" value="127.764" start="0" end="22"/>')
		f.write('\t</node>')

	f.write('</graph>')



