
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

def node_measFunc(G):
	meas = dict()
	bc = nx.betweenness_centrality(G,weight='weight')
	for n in G.nodes():
		meas[n,'sp-betw'] = bc[n]
		meas[n,'mean_oth'] = statistics.mean(bc.values())
	
	return meas

if __name__ == "__main__":
	#nodes = ['a','b','c','d','e']
	nodes = list(range(50))
	ts = list(map(lambda x: x*x, range(2,10,2)))
	tdf = pd.DataFrame(index=ts, columns=['serial','parallel'])

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

		#print('Measuring Graph Properties')
		#t0 = time.time()
		#df = Gt.measGraph(graph_measFunc)
		#tf = time.time()
		#tdf.loc[T,'Graph'] = tf-t0
		#print('took %f seconds.' % (tf-t0,))

		print('Measuring Node Properties Sequentially')
		t0 = time.time()
		ddf = Gt.measNodes(node_measFunc)
		tf = time.time()
		tdf.loc[T,'serial'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('Measuring Node Properties In Paralell')
		t0 = time.time()
		ddf = Gt.measNodes(node_measFunc, parallel=True)
		tf = time.time()
		tdf.loc[T,'parallel'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('EORun\r\n')

	print(tdf)

	plt.plot(df.index,df['meanconstraint'])
	#plt.show()
