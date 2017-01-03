
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from NetTS import *
import random
import statistics
import time


def ut_getsetedges():
	nodes = ['a','b','c','d','e']
	#nodes = list(range(50))
	
	##ts = list(map(lambda x: x*x, range(2,6,2)))
	ts = range(20)
	N = len(ts)
	print('Running for %d iterations.' % len(ts))

	Gt = NetTS(ts,nodes=nodes)
	
	print('Adding Complete Edges For All t')
	for t in ts:
		for i in range(len(nodes)):
			items = list(range(0,len(nodes)))
			items.remove(i)
			samp = random.sample(items,3 if len(items)>=4 else 1)
			for j in samp:
				Gt.setEdgeAttr(t,'weight',{(nodes[i],nodes[j]):random.uniform(0,200),})

	print('Measuring edge properties.')

	df = Gt.getEdgeAttr()
	print(df)

	Gt.setEdgeAttrDF(df)
	df = Gt.getEdgeAttr()
	print(df)

	print('End of Run\r\n')



if __name__ == '__main__':
	ut_getsetedges()