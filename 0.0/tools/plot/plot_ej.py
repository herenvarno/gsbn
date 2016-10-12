import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
	print("usage : ./"+os.path.basename(__file__)+" <filename>")
	exit(-1)

filename = sys.argv[1]
idx=[]
pj0=[]
ej0=[]
zj0=[]
pj1=[]
ej1=[]
zj1=[]
pj9=[]
ej9=[]
zj9=[]


with open(filename, "r") as f:
	lines = f.read().splitlines()
for k,l in enumerate(lines):
	ej_list=l.split(" ")
	idx.append(int(k))
	pj0.append(float(ej_list[0]))
	ej0.append(float(ej_list[1]))
	zj0.append(float(ej_list[2]))
	pj1.append(float(ej_list[3]))
	ej1.append(float(ej_list[4]))
	zj1.append(float(ej_list[5]))
	pj9.append(float(ej_list[6]))
	ej9.append(float(ej_list[7]))
	zj9.append(float(ej_list[8]))
		
#plt.plot(idx, pj0, c='b')
#plt.plot(idx, ej0, c='b', ls='--')
plt.plot(idx, zj0, c='b', ls=':')
#plt.plot(idx, pj1, c='g')
#plt.plot(idx, ej1, c='g', ls='--')
plt.plot(idx, zj1, c='g', ls=':')
#plt.plot(idx, pj9, c='r')
#plt.plot(idx, ej9, c='r', ls='--')
plt.plot(idx, zj9, c='r', ls=':')
#plt.axis([0, 6, 0, 20])
plt.show()
