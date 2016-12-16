import networkx as nx
import pandas as pd

##### Change Detection Functions #####
def get_value_changes(ds):
	''' Takes a data series and outputs (start,val) pairs - 
	one for each change in the value of the data series.
	'''
	changes = [(ds.index[0],ds[ds.index[0]])]
	for ind in ds.index[1:]:
		if ds[ind] != changes[-1][1]:
			changes.append((ind,ds[ind]))
	return changes


##### File Output Functions #####

def build_xgmml_file(f,ndf,edf):
	''' This function builds the xml file given the file object f,
    a graph df, node df, and edge df. First it will look at when 
	attributes change, and then use that to decide when to add an 
	attribute tag.
	'''
	
	t0 = edf.index[0]
	tf = edf.index[-1]

	f.write(header_str)
	f.write(graph_start_str.format(label='mygraph'))

	for n in list(set([x[0] for x in ndf.columns])):
				
		values = {'label':str(n),'id':str(n),'start':t0,'end':tf}
		f.write(node_start_str.format(**values))

		for attr in [x[1] for x in filter(lambda x:x[0]==n,ndf.columns)]:
			changes = get_value_changes(ndf.loc[:,(n,attr)])
			
			write_attr(f,attr,changes,tf)

		f.write(node_end_str)

	for u,v in list(set([x[:2] for x in edf.columns])):
		
		values = {'label':'(%s,%s)'%(str(u),str(v)),'source':str(u),'target':str(v),'start':t0,'end':tf}
		f.write(edge_start_str.format(**values))

		for attr in [x[2] for x in filter(lambda x:x[:2] == (u,v),edf.columns)]:
			changes = get_value_changes(edf.loc[:,(u,v,attr)])

			write_attr(f,attr,changes,tf)

		f.write(edge_end_str)

	f.write(graph_end_str)

	return

def write_attr(f,attr,changes,tf):
	if type(changes[0][1]) is str:
		typ = 'string'
		val = changes = list(map(lambda x: (x[0],str(x[1])), changes))
	elif type(changes[0][1]) is int or type(changes[0][1]) is float:
		typ = 'real'
		val = changes = list(map(lambda x: (x[0],'{:.2f}'.format(float(x[1]))), changes))
	else:
		print('There was an error with the attribute type of the network timeseries.')
		raise

	for c in range(len(changes[:-1])):
		values = {'name':attr,'type':typ,'value':changes[c][1],'start':changes[c][0],'end':changes[c+1][0]}
		f.write(attr_str.format(**values))
	if len(changes) == 1:
		values = {'name':attr,'type':typ,'value':changes[0][1],'start':changes[0][0],'end':tf}
		f.write(attr_str.format(**values))


##### File Output Strings #####
header_str = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!-- Created using the networkxtimeseries library for python. -->\n\n'''

graph_start_str = '<graph label="{label}" directed="0">\n'
graph_end_str = '</graph>\n'

node_start_str = '\t<node label="{label}" id="{id}" start="{start}" end="{end}">\n'
node_end_str = '\t</node>\n'

edge_start_str = '\t<edge label="{label}" source="{source}" target="{target}" start="{start}" end="{end}">\n'
edge_end_str = '\t</edge>\n'

attr_str = '\t\t<att name="{name}" type="{type}" value="{value}" start="{start}" end="{end}"/>\n'




