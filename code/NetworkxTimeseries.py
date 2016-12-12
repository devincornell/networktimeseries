
import networkx as nx
import numpy as np
import pandas as pd
from itertools import *

class NetTs:
    ''' Network Time Series '''
    
    ### member vars ###
    # self.nts - list of networks representing timeseries
    # self.N - number of graphs in the timeseries
    # self.ts is a timeseries list

    def __str__(self):
        return '<NetTs:type=%s,numnodes=%d,numedges=%d>' % (self.type,len(self.nodes) if self.nodes is not None else -1,len(self.edges) if self.edges is not None else -1)

    def __init__(self, ts, nodes=None, edges=None):
        # ts is a timeseries list
        # nodes is a list of node names
        # edges is a list of edges

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
            print('network type not recognized.')

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

        # set edges
        self.edges = edges
        if edges is not None:
            for i in range(self.N):
                for e in edges:
                    self.nts[i].add_edge(e)

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
        for key,val in edata:
            try:
                self.nts[t].edge[key[0]][key[1]]
            else:
                self.nts[t].add_edge(key)
            self.nts[t].edge[key[0]][key[1]][attrName] = val
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
    def measGraph(self, measFunc, addtnlArgs):
        ''' Returns a dataframe of measurements of the 
        entire graph at each point in time. measFunc should
        return a dictionary with keys as columns that are
        expected to be returned in the output dataframe.
        The index will be the timeseries.
        '''
        df = pd.DataFrame()
        for i in range(self.N):
            result = measFunc(self.nts[i], *addtnlArgs)
            tdf = pd.DataFrame([result,],index=[self.ts[i],])
            df.append(tdf)

        return df

    def measNodes(self, measFunc, addtnlArgs):
        ''' Returns a multiindex dataframe of measurements 
        for all nodes at each point in time. measFunc should
        expect a node name and a graph object and return a 
        dictionary with keys as columns that are
        expected as columns in the output dataframe.
        The index will be the timeseries, columns will be 
        multi-indexed: first by node name then by attribute.
        '''
        attr = measFunc(self.nts[0], *addtnlArgs).keys()
        mi = pd.MultiIndex.from_tuples(list(product(self.nodes,attr)))
        df = pd.DataFrame(index=self.ts,columns=mi)

        for i in range(self.N):
            for n in self.nodes:
                result = measFunc(self.nts[i], *addtnlArgs)
                df.loc[(i,),(n,)] = pd.DataFrame([result,],index=[self.ts[i],])

        return df


