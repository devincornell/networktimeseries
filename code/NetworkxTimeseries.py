
import networkx as nx
import numpy as np
import pandas as pd
import multiprocessing

import sys
from itertools import *

class NetTs:
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
			for i in range(self.N):
				for n in nodes:
					self.nts[i].add_node(n)
		else:
			self.nodes = list()

		# set edges
		self.edges = edges
		if edges is not None:
			for i in range(self.N):
				for e in edges:
					self.nts[i].add_edge(e)
		else:
			self.edges = list()

	##### Set Graph, Node, and Edge Attributes #####
	def setGraphAttr(self, t, attrName, gdata):
		''' Adds an attribute to every graph in the network
		at time t. gdata is a list of attributes to apply.
		'''
		for i in range(self.N):
			self.nts[i].graph[attrName] = gdata[i]

		return

	def setNodeAttr(self, t, attrName, ndata):
		''' Adds an attribute to every edge in the network
		at time t. Name specified by attrName and data given 
		in edata, a dictionary of node->vlaue pairs.
		'''
		for key,val in ndata:
			self.nts[t].node[key][attrName] = val
		return

	def setEdgeAttr(self, t, attrName, edata):
		''' Adds an attribute to every edge in the network
		at time t. Name specified by attrName and data given 
		in edata, a dictionary of edge(tuple)->value pairs.
		'''
		for i,j in edata.keys():
			try:
				self.nts[t].edge[i][j]
			except:
				self.nts[t].add_edge(i,j)
				self.edges.append((i,j))
			self.nts[t].edge[i][j][attrName] = edata[(i,j)]
		return

	##### Modify the Graphs and Return NetTs #####
	def modifyGraphs(self, modFunc):
		''' Returns a NetTs object where each graph has 
		been run through modFunc. modFunc 
		should take a graph and return a modified graph.
		'''
		outNet = NetTs(self.N,self.nodes,self.edges)
		for i in range(self.N):
			outNet.nts[i] = modFunc(self.nts[i])

		return outNet

	##### Measure Properties of Graphs Over Time #####
	def measGraph(self, measFunc, addtnlArgs=list()):
		''' Returns a dataframe of measurements of the 
		entire graph at each point in time. measFunc should
		return a dictionary with keys as columns that are
		expected to be returned in the output dataframe.
		The index will be the timeseries.
		'''
		try: # check out measFunc
			dict(measFunc(self.nts[0], *addtnlArgs))
		except TypeError:
			print('Error in measGraph(): measFunc should return a dict')
			exit()

		df = pd.DataFrame()
		for i in range(self.N):
			result = measFunc(self.nts[i], *addtnlArgs)
			tdf = pd.DataFrame([result,],index=[self.ts[i],])
			df = df.append(tdf)

		return df

	def measNodes(self, measFunc, addtnlArgs=list(), pernode=True, parallel=False):
		''' Returns a multiindex dataframe of measurements for all nodes at each 
		point in time.
		
		If pernode: measFunc should expect a node name and a graph
		object and return a dictionary with keys as columns as attribute names.
		
		If not pernode: measFunc should expect a graph object and return a 
		dictionary with (node,attr) as keys.

		Output: The index will be a timeseries, columns will be multi-indexed: 
		first by node name then by attribute.
		'''
		
		if pernode:
			try: trymeas = measFunc(self.nodes[0],self.nts[0], *addtnlArgs)
			except: print('measFunc threw an error:\r\n', sys.stderr[0]); exit()

			try:
				dict(trymeas)
			except TypeError:
				print('Error in measNodes(): measFunc should return a dict')
				exit()
		else:

			try: trymeas = measFunc(self.nts[0], *addtnlArgs)
			except: print('measFunc threw an error:\r\n', sys.stderr[0]); exit()

			try: dict(trymeas)
			except TypeError: print('Error in measNodes(): measFunc should return a dict'); exit()
			
			try: [list(m) for m in trymeas];
			except TypeError: print('Error in measNodes(): measFunc keys should follow (node,attr).'); exit()

		if pernode: # requires a simpler measFunc
			mi = pd.MultiIndex.from_product([self.nodes,trymeas.keys()],names=['node','attr'])
			df = pd.DataFrame(index=self.ts,columns=mi)
			tdata = [(self.nts[t],n,t,measFunc,addtnlArgs) for n in self.nodes for t in self.ts]

			if not parallel:
				meas = list(map(self.thread_measNodes_pernode, tdata))
			else:
				with multiprocessing.Pool(processes=4) as p:
					meas = p.map(self.thread_measNodes_pernode, tdata)
			for n,t,mdf in meas:
				df.loc[[t,],[n,]] = mdf

		else: # requires a more complicated measFunc
			attr = list(set([x[1] for x in trymeas.keys()]))
			mi = pd.MultiIndex.from_product([self.nodes,attr],names=['node','attr'])
			df = pd.DataFrame(index=self.ts,columns=mi)
			tdata = [(self.nts[t],t,measFunc,addtnlArgs) for t in self.ts]

			if not parallel:
				meas = map(self.thread_measNodes, tdata)
			else:
				with multiprocessing.Pool(processes=4) as p:
					meas = p.map(self.thread_measNodes, tdata)
			for t,mdf in meas:
				df.loc[[t],:] = mdf

		return df

	def thread_measNodes_pernode(self,dat):
		''' This is a thread function that will call measFunc on each
		network in the timeseries for each node. measFunc is responsible 
		for returning a dictionary with attribute keys.
		'''

		G,n,t,measFunc,addtnlArgs = dat

		meas = measFunc(n,G, *addtnlArgs)
		result = {(n,k):v for k,v in meas.items()}
		mi = pd.MultiIndex.from_tuples(result.keys())
		
		return n,t,pd.DataFrame([result,],index=[t,],columns=mi)

	def thread_measNodes(self,dat):
		''' This is a thread function that will call measFunc on each
		network in the timeseries. measFunc is responsible for returning
		a dictionary with (node,attr) keys.
		'''

		G,t,measFunc,addtnlArgs = dat

		meas = measFunc(G, *addtnlArgs)
		#result = {(n,k):v for k,v in meas.items()}
		mi = pd.MultiIndex.from_tuples(meas.keys())
		
		return t,pd.DataFrame([meas,],index=[t,],columns=mi)
