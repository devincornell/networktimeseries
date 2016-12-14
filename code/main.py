
import networkx as nx
import matplotlib.pyplot as plt

import NetworkxTimeseries as nxt
import random
import statistics

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
	nodes = ['a','b','c','d','e']
	#nodes = list(range(100))

	ts = list(range(int(1.0e1)))
	N = len(ts)

	Gt = nxt.NetTs(ts,nodes=nodes)
	
	print('Adding Edges For All Years')
	for t in ts:
		for i in range(len(nodes)):
			for j in range(i,len(nodes)):
				Gt.setEdgeAttr(t,'weight',{(nodes[i],nodes[j]):random.uniform(0,10),})

	print('Measuring Graph Properties')
	df = Gt.measGraph(graph_measFunc)

	print('Measuring Node Properties Per-Node')
	ddf = Gt.measNodes(pernode_measFunc, pernode=True)

	print('Measuring Node Properties At Once')
	ddf = Gt.measNodes(nodes_measFunc, pernode=False)

	print(ddf)

	#plt.plot(df.index,df['meanconstraint'])
	#plt.show()


