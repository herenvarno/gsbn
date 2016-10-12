import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 4:
	print("usage : ./"+os.path.basename(__file__)+" <text dump directory> <parameter> <mcu> <hcu>")
	exit(-1)

subdirs = os.listdir(sys.argv[1])
subdirs.sort(key=lambda x: float(x))
src_mcu=[]
dest_mcu=[]
param=[]

filename = os.path.join(sys.argv[1], str(sys.argv[2]), "ij.txt");
with open(filename, "r") as f:
	lines = f.read().splitlines()
for l in lines:
	l=l.split(" ")
	src_mcu.append(int(l[0]))
	dest_mcu.append(int(l[1]))
	if sys.argv[3] == "pij":
		param.append(float(l[2]))
	elif sys.argv[3] == "eij":
		param.append(float(l[3]))
	elif sys.argv[3] == "zi2":
		param.append(float(l[4]))
	elif sys.argv[3] == "zj2":
		param.append(float(l[5]))
	elif sys.argv[3] == "tij":
		param.append(float(l[6]))
	elif sys.argv[3] == "wij":
		param.append(float(l[7]))
		
plt.scatter(dest_mcu, src_mcu, c=param)
plt.xlabel("MCU index (j)")
plt.ylabel("MCU index (i)")
plt.xlim([-10, 110])
plt.ylim([-10, 110])
#plt.axis([0, 6, 0, 20])
plt.show()
