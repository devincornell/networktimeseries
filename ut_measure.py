
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from NetTS import *
import random
import statistics
import time

def graph_measFunc(G):
	return {
		'meanconstraint':statistics.mean(nx.get_edge_attributes(G,'weight').values())
		}

def node_measFunc(G):
	meas = dict()
	bc = nx.betweenness_centrality(G,weight='weight')
	for n in G.nodes():
		meas[n,'sp-betw'] = bc[n]
		meas[n,'mean_oth'] = statistics.mean(bc.values())
	
	return meas

def edge_measFunc(G):
	meas = dict()
	nodes = list(G.nodes())
	for u in nodes:
		for v in nodes[nodes.index(u):]:
			meas[(u,v,'weight')] = G.get_edge_data(u,v,'weight')
	
	return meas

def ut_measure():
	nodes = ['a','b','c','d','e']
	#nodes = list(range(50))
	#ts = list(map(lambda x: x*x, range(2,6,2)))
	ts = [50000,]
	tdf = pd.DataFrame(index=ts, columns=['graph','nodes', 'edges'])

	for T in ts:

		print('Running for T = %d.' % T)
		ts = list(range(T))
		N = len(ts)

		Gt = NetTS(ts,nodes=nodes)
	
		print('Adding Complete Edges For All t')
		for t in ts:
			for i in range(len(nodes)):
				for j in range(i,len(nodes)):
					Gt.setEdgeAttr(t,'weight',{(nodes[i],nodes[j]):random.uniform(0,10),})


		print('Measuring Graph Properties in Parallel')
		t0 = time.time()
		df = Gt.measure(graph_measFunc, meas_obj='graph', parallel=True)
		tf = time.time()
		tdf.loc[T,'graph'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('Measuring Node Properties In Parallel')
		t0 = time.time()
		df = Gt.measure(node_measFunc, meas_obj='nodes', parallel=True)
		tf = time.time()
		tdf.loc[T,'nodes'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('Measuring Edge Properties In Parallel')
		t0 = time.time()
		ddf = Gt.measure(edge_measFunc, meas_obj='edges', parallel=True)
		tf = time.time()
		tdf.loc[T,'edges'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('End of Run\r\n')


	print(tdf)


if __name__ == '__main__':
	ut_measure()