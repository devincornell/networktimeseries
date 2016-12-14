
# system imports
import multiprocessing
import pickle
import sys
from itertools import *


# anaconda imports
import networkx as nx
import numpy as np
import pandas as pd

# local imports
import xgmml

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
		gdf = self.measure(xgmml.meas_graph_attr, meas_obj='graph', parallel=False)
		ndf = self.measure(xgmml.meas_node_attr, meas_obj='nodes', parallel=False)
		edf = self.measure(xgmml.meas_edge_attr, meas_obj='edges', parallel=False)
		with open(filename) as f:
			xgmml.build_xgmml_file(f,gdf,ndf,edf)

		return

	def __init__(self, ts, nodes=None, edges=None):
		ts = list(ts) # ts is a timeseries list
		if nodes is not None: nodes = list(nodes) # nodes is a list of node names
		if edges is not None: edges = list(edges) # edges is a list of edges

		# set timeseries type
		if nodes is None:
			self.type = 'nonstatic'
			print('Error - choose at least a set of nodes in NetTs init.')
			exit()

		elif nodes is not None and edges is None:
			self.type = 'static_nodes'
		elif nodes is not None and edges is not None:
			self.type = 'static_structure'
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

	##### Modify the Graphs and Return NetTs #####
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
	def measure(self, measFunc, meas_obj='graph', addtnlArgs=list(), parallel=False):
		''' Returns a multiindex dataframe of measurements for all nodes at each 
		point in time. measFunc should expect a graph object and return a 
		dictionary with (node,attr) as keys. Output: The index will be a timeseries, 
		columns will be multi-indexed - first by node name then by attribute.
		'''
		# error checking
		if not (meas_obj == 'graph' or meas_obj == 'nodes' or meas_obj == 'edges'): raise
		try: trymeas = measFunc(self.nts[0], *addtnlArgs)
		except: print('measFunc threw an error:\r\n', sys.stderr[0]); exit()
		try: dict(trymeas)
		except TypeError: print('Error in measure(): measFunc should return a dict'); exit()
		if meas_obj == 'nodes' or meas_obj == 'edges':
			try: [list(m) for m in trymeas];
			except TypeError: print('Error in measure(): measFunc keys should follow (node,attr).'); exit()

		if meas_obj == 'graph':
			cols = list(trymeas.keys())
		elif meas_obj == 'nodes':
			cols = pd.MultiIndex.from_tuples(trymeas.keys(),names=['node','attr'])
		elif meas_obj == 'edges':
			cols = pd.MultiIndex.from_tuples(trymeas.keys(),names=['from','to','attr'])

		df = pd.DataFrame(index=self.ts,columns=cols)
		tdata = [(self[t],t,measFunc,addtnlArgs,meas_obj,cols) for t in self.ts]

		if not parallel:
			meas = map(self.thread_measure, tdata)
		else:
			with multiprocessing.Pool(processes=4) as p:
				meas = p.map(self.thread_measure, tdata)

		for t,mdf in meas:
			df.loc[[t],:] = mdf

		return df

	def thread_measure(self, dat):
		''' This is a thread function that will call measFunc on each
		network in the timeseries. measFunc is responsible for returning
		a dictionary with (node,attr) keys.
		'''
		G,t,measFunc,addtnlArgs,meas_obj,cols = dat
		meas = measFunc(G, *addtnlArgs)
		
		return t,pd.DataFrame([meas,],index=[t,],columns=cols)

