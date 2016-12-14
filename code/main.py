﻿
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import NetworkxTimeseries as nxt
import random
import statistics

import time

def graph_measFunc(G):
	return {
		'meanconstraint':statistics.mean(nx.get_edge_attributes(G,'weight').values())
		}

def pernode_measFunc(n, G):
	bc = nx.betweenness_centrality(G,weight='weight')
	return {'sp-betweeness':bc[n],'others':statistics.mean(bc.values())}

def nodes_measFunc(G):
	meas = dict()
	bc = nx.betweenness_centrality(G,weight='weight')
	for n in G.nodes():
		meas[n,'sp-betw'] = bc[n]
		meas[n,'mean_oth'] = statistics.mean(bc.values())
	
	return meas

if __name__ == "__main__":
	#nodes = ['a','b','c','d','e']
	nodes = list(range(50))
	ts = list(map(lambda x: x*x, range(10,1000, 100)))
	tdf = pd.DataFrame(index=ts, columns=['Graph','nodewise','nodes'])

	for T in ts:

		print('Running for T = %d.' % T)
		ts = list(range(T))
		N = len(ts)

		Gt = nxt.NetTs(ts,nodes=nodes)
	
		print('Adding Complete Edges For All t')
		for t in ts:
			for i in range(len(nodes)):
				for j in range(i,len(nodes)):
					Gt.setEdgeAttr(t,'weight',{(nodes[i],nodes[j]):random.uniform(0,10),})

		print('Measuring Graph Properties')
		t0 = time.time()
		df = Gt.measGraph(graph_measFunc)
		tf = time.time()
		tdf.loc[T,'Graph'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('Measuring Node Properties Per-Node')
		t0 = time.time()
		ddf = Gt.measNodes(pernode_measFunc, pernode=True)
		tf = time.time()
		tdf.loc[T,'nodewise'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('Measuring Node Properties At Once')
		t0 = time.time()
		ddf = Gt.measNodes(nodes_measFunc, pernode=False)
		tf = time.time()
		tdf.loc[T,'nodes'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

	print(ddf)

	plt.plot(df.index,df['meanconstraint'])
	#plt.show()