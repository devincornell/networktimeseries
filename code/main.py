
import networkx as nx
import matplotlib.pyplot as plt

import NetworkxTimeseries as nxt
import random
import statistics

def measFunc(G):
    return {
        'meanconstraint':statistics.mean(nx.get_edge_attributes(G,'weight').values())
        }


if __name__ == "__main__":
    nodes = ['a','b','c','d','e']

    ts = range(100)
    N = len(ts)

    Gt = nxt.NetTs(ts,nodes=nodes)
    
    for t in ts:
        for i in range(N):
            for j in range(N-i):
                Gt.setEdgeAttr(t,'weight',{(i,j):random.uniform(0,10),})

    df = Gt.measGraph(measFunc)
    print(df)

    plt.plot(df.index,df['meanconstraint'])
    plt.show()
