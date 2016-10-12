import sys
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

if len(sys.argv) < 6:
	print("usage : ./"+os.path.basename(__file__)+" <text dump directory> <parameter> <mcu 1> <mcu 2> <mcu 3>")
	exit(-1)

subdirs = os.listdir(sys.argv[1])
subdirs.sort(key=lambda x: float(x))
t1=[]
param1=[]
t2=[]
param2=[]
t3=[]
param3=[]
color='r'

for d in subdirs:
	filename = os.path.join(sys.argv[1], d, "j.txt");
	print(filename)
	with open(filename, "r") as f:
		lines = f.read().splitlines()
	for l in lines:
		l=l.split(" ")
		mcu = l[0];
		if(True):
			if sys.argv[2]=="pj":
				t1.append(float(d))
				param1.append(float(l[1]))
			elif sys.argv[2]=="ej":
				t1.append(float(d))
				param1.append(float(l[2]))
			elif sys.argv[2]=="zj":
				t1.append(float(d))
				param1.append(float(l[3]))
			elif sys.argv[2]=="bj":
				t1.append(float(d))
				param1.append(float(l[4]))
			elif sys.argv[2]=="epsc":
				t1.append(float(d))
				param1.append(float(l[5]))

if sys.argv[2]=="pj":
	color='r'
elif sys.argv[2]=="ej":
	color='violet'
elif sys.argv[2]=="zj":
	color='g'
elif sys.argv[2]=="bj":
	color='b'
elif sys.argv[2]=="epsc":
	color='y'


param1 = np.asarray(param1)
param1 = param1
mcu=[]
for i in range(int(len(t1)/100)):
	for j in range(100):
		mcu.append(float(j))

print(len(mcu));

fig = plt.figure()
ax = fig.gca(projection='3d')
print(len(t1));
print(param1.shape)
surf = ax.plot_surface(t1, mcu, param1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
