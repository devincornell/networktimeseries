
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import NetworkxTimeseries as nxt
import random
import statistics

import time

def ut_ntsfiles():
	#nodes = ['a','b','c','d','e']
	nodes = list(range(50))
	ts = list(map(lambda x: x*x, range(2,6,2)))
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

		fname = 'T_%d.nts' % (T,)
		print('Saving nts file:', Gt)
		Gt.save_nts(fname)

		nts = nxt.NetTS.open_nts(fname)
		print('Opened nts file:', nts)

	#plt.plot(df.index,df['meanconstraint'])
	#plt.show()
