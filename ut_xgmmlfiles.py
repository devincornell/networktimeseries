
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from NetTS import *
import random
import statistics
import time

def ut_xgmmlfiles():
	#nodes = ['a','b','c','d','e']
	nodes = list(range(50))
	
	#ts = list(map(lambda x: x*x, range(2,6,2)))
	ts = range(200)
	N = len(ts)
	print('Running for %d iterations.' % len(ts))

	Gt = NetTS(ts,nodes=nodes)
	
	print('Adding Complete Edges For All t')
	for t in ts:
		for i in range(len(nodes)):
			for j in range(i,len(nodes)):
				Gt.setEdgeAttr(t,'weight',{(nodes[i],nodes[j]):random.uniform(0,200),})

	print('Saving File')
	Gt.save_xgmml('test.xgmml')

	print('End of Run\r\n')


if __name__ == '__main__':
	ut_xgmmlfiles()