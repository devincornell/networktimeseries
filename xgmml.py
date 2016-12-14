import networkx as nx

##### Measurement Functions #####
def meas_graph_attr(G):
	meas=dict(G.graph)
	return meas

def meas_node_attr(G):
	meas = dict()
	nx.get_node_attributes(G,G.nodes()[0])
	for a,val in attr.values():
		attr = nx.get_node_attributes(G,a)
		meas.update({(n,a):val for n in G.nodes()})
	
	return meas

def meas_edge_attr(G):
	meas = dict()
	nx.get_edge_attributes(G,G.edges()[0])
	for a,val in attr.values():
		attr = nx.get_node_attributes(G,a)
		meas.update({(e[0],e[1],a):val for e in G.edges()})

	return meas

##### Change Detection Functions #####



##### File Output Functions #####
def build_xgmml_file(f,gdf,ndf,edf):
	''' This function builds the xml file given the file object f,
    a graph df, node df, and edge df. First it will look at when 
	attributes change, and then use that to decide when to add an 
	attribute tag.
	'''
	pass