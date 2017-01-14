
# system imports
import multiprocessing
import pickle
import sys
from itertools import *


# anaconda imports
import networkx as nx
import numpy as np
import pandas as pd


class NetTS:
	''' Network Time Series '''
	
	### member vars ###
	# self.nts - list of networks representing timeseries
	# self.N - number of graphs in the timeseries
	# self.ts is a timeseries list

	def __str__(self):
		return '<NetTs:type=%s,numnodes=%d,numedges=%d>' % (
			self.type,
			len(self.nodes) if self.nodes is not None else -1,
			len(self.edges) if self.edges is not None else -1
			)

	def __getitem__(self,key):
		i = self.ts.index(key)
		return self.nts[i]

	@staticmethod
	def open_nts(ntsfile):
		data = None
		with open(ntsfile,'rb') as f:
			data = pickle.load(f)
		return data

	def save_nts(self,ntsfile):
		with open(ntsfile,mode='wb') as f:
			data = pickle.dump(self,f)
		return

	def save_xgmml(self, filename):
		self.update()
		ndf = self.getNodeAttr()
		edf = self.getEdgeAttr()
		with open(filename,'w') as f:
			build_xgmml_file(f,ndf,edf)

		return

	def __init__(self, ts, nodes=None, edges=None, type='static_nodes'):
		ts = list(ts) # ts is a timeseries list
		if nodes is not None: nodes = list(nodes) # nodes is a list of node names
		if edges is not None: edges = list(edges) # edges is a list of edges

		# set timeseries type
		if type == 'static_nodes' or type == 'static_structure':
			self.type = type
		elif type == 'dynamic':
			print('Error - choose at least a set of nodes in NetTS init.')
			print('Support for dynamic nodes is not supported.')
			exit()
		else:
			print('network type not recognized in NetTs init.')
			exit()

		# make networks
		self.ts = ts
		self.N = len(ts)
		self.nts = []
		for i in range(self.N):
			self.nts.append(nx.Graph(name=ts[i]))

		# set nodes
		self.nodes = nodes
		if nodes is not None:
			for t in self.ts:
				for n in nodes:
					self[t].add_node(n)
		else:
			self.nodes = list()

		# set edges
		self.edges = edges
		if edges is not None:
			for t in self.ts:
				for e in edges:
					self[t].add_edge(e)
		else:
			self.edges = list()

	def update(self):
		''' This function will add _tag attributes to every node/edge and also
		keep track of all nodes/edges that appear at all times.'''
		
		# update self.nodes, self.edges to contain all possible unique edges
		nodeset = set()
		edgeset = set()
		for t in self.ts:
			nodeset.union(self[t].nodes())
			edgeset.union(self[t].edges())
		self.nodes = list(nodeset)
		self.edges = list(edgeset)

		# ensure every node/edge has a _tag attribute
		for t in self.ts:
			nx.set_node_attributes(self[t],'_tag',{n:str(n) for n in self[t].nodes()})
			nx.set_edge_attributes(self[t],'_tag',{e:str(e) for e in self[t].edges()})

	##### Set Graph, Node, and Edge Attributes #####
	def setGraphAttr(self, t, attrName, gdata):
		''' Adds an attribute to every graph in the network
		at time t. gdata is a list of attributes to apply.
		'''
		for t in self.ts:
			self[t].graph[attrName] = gdata[i]
		return

	def setNodeAttr(self, t, attrName, ndata):
		''' Adds an attribute to every edge in the network
		at time t. Name specified by attrName and data given 
		in edata, a dictionary of node->vlaue pairs.
		'''
		for key,val in ndata:
			self[t].node[key][attrName] = val
		return

	def setEdgeAttr(self, t, attrName, edata):
		''' Adds an attribute to every edge in the network
		at time t. Name specified by attrName and data given 
		in edata, a dictionary of edge(tuple)->value pairs.
		'''
		for i,j in edata.keys():
			try:
				self[t].edge[i][j]
			except:
				self[t].add_edge(i,j)
				self.edges.append((i,j))
			self[t].edge[i][j][attrName] = edata[(i,j)]
		return

	def setEdgeAttrDF(self, df):
		''' Adds edge data assuming that edata is a pandas
		dataframe formatted with multiindexed columns
		(u,v,attr) and indexed rows with time.
		'''
		for u in mdf(df.columns,()):
			for v in mdf(df.columns,(u,)):
				for attr in mdf(df.columns,(u,v)):
					for t in df.index:
						try:
							self[t].edge[u][v]
						except KeyError:
							self[t].add_edge(u,v)

						self[t].edge[u][v][attr] = df.loc[t,(u,v,attr)]


	##### Modify the Graphs and Return NetTS #####
	def modifyGraphs(self, modFunc):
		''' Returns a NetTs object where each graph has 
		been run through modFunc. modFunc 
		should take a graph and return a modified graph.
		'''
		outNet = NetTs(self.ts,nodes=self.nodes,edges=self.edges)
		for t in self.ts:
			outNet[t] = modFunc(self[t])

		return outNet

	##### Measure Properties of Graphs Over Time #####
	def time_measure(self, measFunc, meas_obj='graph', addtnlArgs=list(), parallel=False):
		''' Returns a multiindex dataframe of measurements for all nodes at each 
		point in time. measFunc should expect a graph object and return a 
		dictionary with (node,attr) as keys. Output: The index will be a timeseries, 
		columns will be multi-indexed - first by node name then by attribute.
		'''
		# error checking
		if not (meas_obj == 'graph' or meas_obj == 'nodes' or meas_obj == 'edges'): raise
		trymeas = measFunc(self.nts[0], *addtnlArgs)
		try: dict(trymeas)
		except TypeError: print('Error in measure(): measFunc should return a dict'); exit()
		if meas_obj == 'nodes' or meas_obj == 'edges':
			try: [list(m) for m in trymeas];
			except TypeError: print('Error in measure(): measFunc keys should follow (node,attr).'); exit()

		if len(trymeas) == 0: # return empty dataframe
			return pd.DataFrame()

		if meas_obj == 'graph':
			cols = list(trymeas.keys())
		elif meas_obj == 'nodes':
			cols = pd.MultiIndex.from_tuples(trymeas.keys(),names=['node','attr'])
		elif meas_obj == 'edges':
			cols = pd.MultiIndex.from_tuples(trymeas.keys(),names=['from','to','attr'])

		df = pd.DataFrame(index=self.ts,columns=cols)
		tdata = [(self[t],t,measFunc,addtnlArgs,meas_obj,cols) for t in self.ts]

		if not parallel:
			meas = map(self.thread_time_measure, tdata)
		else:
			with multiprocessing.Pool(processes=4) as p:
				meas = p.map(self.thread_time_measure, tdata)
		for t,mdf in meas:
			df.loc[[t],:] = mdf

		return df

	def thread_time_measure(self, dat):
		''' This is a thread function that will call measFunc on each
		network in the timeseries. measFunc is responsible for returning
		a dictionary with (node,attr) keys.
		'''
		G,t,measFunc,addtnlArgs,meas_obj,cols = dat
		meas = measFunc(G, *addtnlArgs)
		
		return t,pd.DataFrame([meas,],index=[t,],columns=cols)

	def getNodeAttr(self,t=None,parallel=False):
		''' Measure all node attributes across time.
		'''
		ndf = self.time_measure(meas_node_attr, meas_obj='nodes', parallel=parallel)
		ndf.sort_index(axis='columns',inplace=True)
		return ndf

	def getEdgeAttr(self,t=None,parallel=False):
		''' Measure all edge attributes across time.
		'''
		edf = self.time_measure(meas_edge_attr, meas_obj='edges', parallel=parallel)
		edf.sort_index(axis='columns',inplace=True)
		return edf

def mdf(mi,match):
    ''' Returns the list of children of the ordered match
    set given by match. Specifically for dataframe looping.
	'''
    matchfilt = filter(lambda x: x[:len(match)] == match,mi)
    return set([x[len(match)] for x in matchfilt])


##### Standalone Measurement Functions #####
''' These functions are used in the class but not explicitly class 
members.
'''

def meas_node_attr(G):
	meas = dict()
	attrnames = G.nodes(data=True)[0][1].keys() # attr dict from first node
	for attrname in attrnames:
		attr = nx.get_node_attributes(G,attrname)
		meas.update({(n,attrname):attr[n] for n in G.nodes()})

	return meas

def meas_edge_attr(G):
	meas = dict()
	e0 = G.edges()[0]
	attrnames = G.get_edge_data(e0[0],e0[1]).keys()
	for attrname in attrnames:
		attr = nx.get_edge_attributes(G,attrname)
		meas.update({(e[0],e[1],attrname):attr[e] for e in G.edges()})

	return meas



##### Change Detection Functions #####
def get_value_changes(ds):
	''' Takes a data series and outputs (start,val) pairs - 
	one for each change in the value of the data series.
	'''
	changes = [(ds.index[0],ds[ds.index[0]])]
	for ind in ds.index[1:]:
		if ds[ind] != changes[-1][1]:
			changes.append((ind,ds[ind]))
	return changes


##### XGMML File Output Functions #####

def build_xgmml_file(f,ndf,edf):
	''' This function builds the xml file given the file object f,
    a graph df, node df, and edge df. First it will look at when 
	attributes change, and then use that to decide when to add an 
	attribute tag.
	'''
	
	t0 = edf.index[0]
	tf = edf.index[-1]

	f.write(header_str)
	f.write(graph_start_str.format(label='mygraph'))

	for n in list(set([x[0] for x in ndf.columns])):
				
		values = {'label':str(n),'id':str(n),'start':t0,'end':tf}
		f.write(node_start_str.format(**values))

		for attr in [x[1] for x in filter(lambda x:x[0]==n,ndf.columns)]:
			changes = get_value_changes(ndf.loc[:,(n,attr)])
			
			write_attr(f,attr,changes,tf)

		f.write(node_end_str)

	for u,v in list(set([x[:2] for x in edf.columns])):
		
		values = {'label':'(%s,%s)'%(str(u),str(v)),'source':str(u),'target':str(v),'start':t0,'end':tf}
		f.write(edge_start_str.format(**values))

		for attr in [x[2] for x in filter(lambda x:x[:2] == (u,v),edf.columns)]:
			changes = get_value_changes(edf.loc[:,(u,v,attr)])

			write_attr(f,attr,changes,tf)

		f.write(edge_end_str)

	f.write(graph_end_str)

	return

def write_attr(f,attr,changes,tf):

	if type(changes[0][1]) is str:
		typ = 'string'
		changes = list(map(lambda x: (x[0],str(x[1])), changes))
	elif type(changes[0][1]) is int or type(changes[0][1]) is float:
		typ = 'real'
		changes = list(map(lambda x: (x[0],'{:.9f}'.format(float(x[1]))), changes))
	else:
		print('There was an error with the attribute type of the network timeseries.')
		raise

	for i in range(len(changes[:-1])):
		if changes[i][1] is not 'nan' and changes[i][1] is not 'None':
			values = {'name':attr,'type':typ,'value':changes[i][1],'start':changes[i][0],'end':changes[i+1][0]}
			f.write(attr_str.format(**values))
	if len(changes) == 1 and changes[0][1] is not 'None' and changes[0][1] is not 'nan':
		values = {'name':attr,'type':typ,'value':changes[0][1],'start':changes[0][0],'end':tf}
		f.write(attr_str.format(**values))


##### File Output Strings #####
header_str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!-- Created using the networkxtimeseries library for python. -->\n\n'''

graph_start_str = '<graph label="{label}" directed="0">\n'
graph_end_str = '</graph>\n'

node_start_str = '\t<node label="{label}" id="{id}" start="{start}" end="{end}">\n'
node_end_str = '\t</node>\n'

edge_start_str = '\t<edge label="{label}" source="{source}" target="{target}" start="{start}" end="{end}">\n'
edge_end_str = '\t</edge>\n'

attr_str = '\t\t<att name="{name}" type="{type}" value="{value}" start="{start}" end="{end}"/>\n'



