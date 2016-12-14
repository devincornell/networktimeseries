
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

if __name__ == "__main__":
	#nodes = ['a','b','c','d','e']
	nodes = list(range(50))
	ts = list(map(lambda x: x*x, range(2,26,2)))
	tdf = pd.DataFrame(index=ts, columns=['serial','parallel'])

	for T in ts:

		print('Running for T = %d.' % T)
		ts = list(range(T))
		N = len(ts)

		Gt = nxt.NetTS(ts,nodes=nodes)
	
		print('Adding Complete Edges For All t')
		for t in ts:
			for i in range(len(nodes)):
				for j in range(i,len(nodes)):
					Gt.setEdgeAttr(t,'weight',{(nodes[i],nodes[j]):random.uniform(0,10),})

		print('Measuring Graph Properties in Serial')
		t0 = time.time()
		df = Gt.measGraph(graph_measFunc, parallel=False)
		tf = time.time()
		tdf.loc[T,'serial'] = tf-t0
		print('took %f seconds.' % (tf-t0,))

		print('Measuring Graph Properties in Parallel')
		t0 = time.time()
		df = Gt.measGraph(graph_measFunc, parallel=True)
		tf = time.time()
		tdf.loc[T,'parallel'] = tf-t0
		print('took %f seconds.' % (tf-t0,))


		#Gt.save_nts('T_%d.nts' % (T,))


	print(tdf)

	#plt.plot(df.index,df['meanconstraint'])
	#plt.show()
