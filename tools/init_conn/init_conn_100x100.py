#!/usr/bin/env python
import os
import sys
import random
import math
import networkx as nx

import matplotlib.pyplot as plt

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or PyDotPlus")

if len(sys.argv) < 3:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <local connectivity rate> <output file>")
	exit(-1)

LOCAL = float(sys.argv[1])
GLOBAL = 1-LOCAL

HEIGHT = 20
WIDTH = 20

WL = 2
WR = 2
WU = 2
WD = 2

WINDOW_HEIGHT = WU+WD+1
WINDOW_WIDTH = WL+WR+1

DIM_HCU = HEIGHT * WIDTH
DIM_MCU = 10
DIM_CONN = (WINDOW_HEIGHT * 2 - 1) * (WINDOW_WIDTH * 2 -1) * DIM_MCU

G_LOCAL = nx.DiGraph()
G_GLOBAL = nx.DiGraph()
pos = {}
for x in range(HEIGHT):
	for y in range(WIDTH):
		idx_src = x*WIDTH+y
		pos[idx_src] = (x, y)
		G_LOCAL.add_node(idx_src)
		G_GLOBAL.add_node(idx_src)

# LOCAL CONNECTION
for x in range(HEIGHT):
	for y in range(WIDTH):
		idx_src = x*WIDTH+y
		pos[idx_src] = (x, y)
		active_local_idx = [r for r in range(WINDOW_HEIGHT*WINDOW_WIDTH)]
		random.shuffle(active_local_idx)
		active_local_idx=active_local_idx[:int(LOCAL*WINDOW_HEIGHT*WINDOW_WIDTH)]
		ii=0
		for xx in range(x-WU, x+WD+1):
			for yy in range(y-WL, y+WR+1):
				ii += 1
				if xx>=0 and yy>=0 and xx<HEIGHT and yy<WIDTH:
					idx_dest = xx*WIDTH + yy
					if ii-1 in active_local_idx:
						G_LOCAL.add_edge(idx_src, idx_dest)
		

# GLOBAL CONNECTION
avail_node = [[i,0]  for i in range(HEIGHT*WIDTH)]
for x in range(HEIGHT):
	for y in range(WIDTH):
		idx_dest = x*WIDTH+y
		for xx in range(x - WINDOW_HEIGHT + 1, x + WINDOW_HEIGHT -1):
			for yy in range(y - WINDOW_WIDTH + 1, y + WINDOW_WIDTH -1):
				if xx>=0 and yy>=0 and xx<HEIGHT and yy<WIDTH:
					idx_src = xx*WIDTH+yy
					if not G_LOCAL.has_edge(idx_src, idx_dest):
						avail_node[idx_dest][1] += 1

for x in range(HEIGHT):
	for y in range(WIDTH):
		idx_src = x*WIDTH+y
		random.shuffle(avail_node)
		l=[]
		for i in range(len(avail_node)):
			if i<GLOBAL*WINDOW_HEIGHT*WINDOW_WIDTH:
				G_GLOBAL.add_edge(idx_src, avail_node[i][0])
				avail_node[i][1] -= 1
			if avail_node[i][1]>0 :
				l.append(avail_node[i])
		avail_node=l

#print(G_LOCAL.out_degree(0),G_GLOBAL.out_degree(0))
#print(G_LOCAL.out_degree(),G_GLOBAL.out_degree())

plt.figure(figsize=(8,8))
if(GLOBAL>LOCAL):
	plt.title("Connections of BCPNN (20x20 HCUs)\n RED=Global connection; BLUE=Local connection\n GLOBAL:LOCAL = "+"{:.2f}".format(GLOBAL/LOCAL)+":1")
else:
	plt.title("Connections of BCPNN (20x20 HCUs)\n RED=Global connection; BLUE=Local connection\n GLOBAL:LOCAL = 1:"+"{:.2f}".format(LOCAL/GLOBAL))
nx.draw(G_GLOBAL, pos, node_size=20, alpha = 0.5, node_color="black", edge_color="red", with_labels=False)
nx.draw(G_LOCAL, pos, node_size=20, alpha = 0.5, node_color="black", edge_color="blue", with_labels=False)
plt.show()

with open(sys.argv[2], "w+") as f:
	for idx_dest in range(HEIGHT*WIDTH):
		f.write("0,"+str(idx_dest))
		for idx_src in range(HEIGHT*WIDTH):
			if G_LOCAL.has_edge(idx_src, idx_dest):
				x = idx_src/WIDTH
				y = idx_src%WIDTH
				xx = idx_dest/WIDTH
				yy = idx_dest%WIDTH
				d = int(round(math.sqrt((x-xx)*(x-xx)+(y-yy)*(y-yy)))) + 1
				for c in range(DIM_MCU):
					f.write(","+str(idx_src*DIM_MCU+c)+","+str(d))
		f.write('\n')
