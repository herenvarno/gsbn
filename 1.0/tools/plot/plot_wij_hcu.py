import sys
import os
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

if len(sys.argv) < 3:
	print("usage : ./"+os.path.basename(__file__)+" <text dump directory> <timestamp>")
	exit(-1)

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
	param.append(float(l[7]))
	mat[int(l[0])][int(l[1])]=float(l[7])



fig, ax = plt.subplots()
cax = ax.imshow(mat,interpolation='nearest', cmap=cm.coolwarm)
ax.set_title('Wij')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax)



plt.show()
