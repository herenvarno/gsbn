import sys
import os
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

if len(sys.argv) < 4:
	print("usage : ./"+os.path.basename(__file__)+" <text dump directory> <timestamp> <parameter>")
	exit(-1)

subdirs = os.listdir(sys.argv[1])
subdirs.sort(key=lambda x: float(x))
src_mcu=[]
dest_mcu=[]
param=[]
mat = np.zeros([100, 100])
filename = os.path.join(sys.argv[1], str(sys.argv[2]), "ij.txt");
with open(filename, "r") as f:
	lines = f.read().splitlines()
for l in lines:
	l=l.split(" ")
	src_mcu.append(int(l[0]))
	dest_mcu.append(int(l[1]))
	if sys.argv[3] == "pij":
		param.append(float(l[2]))
		mat[int(l[0])][int(l[1])]=float(l[2])
	elif sys.argv[3] == "eij":
		param.append(float(l[3]))
		mat[int(l[0])][int(l[1])]=float(l[3])
	elif sys.argv[3] == "zi2":
		param.append(float(l[4]))
		mat[int(l[0])][int(l[1])]=float(l[4])
	elif sys.argv[3] == "zj2":
		param.append(float(l[5]))
		mat[int(l[0])][int(l[1])]=float(l[5])
	elif sys.argv[3] == "tij":
		param.append(float(l[6]))
		mat[int(l[0])][int(l[1])]=float(l[6])
	elif sys.argv[3] == "wij":
		param.append(float(l[7]))
		mat[int(l[0])][int(l[1])]=float(l[7])


fig, ax = plt.subplots()
cax = ax.imshow(mat,interpolation='nearest', cmap=cm.seismic)
ax.set_title(sys.argv[3])
# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax)
plt.show()
