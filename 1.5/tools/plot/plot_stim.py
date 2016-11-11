import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
	print("usage : ./"+os.path.basename(__file__)+" <text dump directory>")
	exit(-1)

subdirs = os.listdir(sys.argv[1])
subdirs.sort(key=lambda x: float(x))
t=[]
idx=[]
stim=[]

for d in subdirs:
	filename = os.path.join(sys.argv[1], d, "stim.txt");
	print(filename)
	with open(filename, "r") as f:
		lines = f.read().splitlines()
	for k,l in enumerate(lines):
		if(float(l)>0):
			t.append(float(d))
			idx.append(int(k))
			stim.append(float(l))
		
plt.scatter(t, idx, c=stim)
plt.xlabel("time [s]")
plt.ylabel("MCU index")
plt.xlim([-1, 3])
plt.ylim([-10, 110])
#plt.axis([0, 6, 0, 20])
plt.show()
