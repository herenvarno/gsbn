import sys
import os
import re
import matplotlib.pyplot as plt
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
		if(mcu==sys.argv[3]):
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
		elif(mcu==sys.argv[4]):
			if sys.argv[2]=="pj":
				t2.append(float(d))
				param2.append(float(l[1]))
			elif sys.argv[2]=="ej":
				t2.append(float(d))
				param2.append(float(l[2]))
			elif sys.argv[2]=="zj":
				t2.append(float(d))
				param2.append(float(l[3]))
			elif sys.argv[2]=="bj":
				t2.append(float(d))
				param2.append(float(l[4]))
			elif sys.argv[2]=="epsc":
				t2.append(float(d))
				param2.append(float(l[5]))
		elif(mcu==sys.argv[5]):
			if sys.argv[2]=="pj":
				t3.append(float(d))
				param3.append(float(l[1]))
			elif sys.argv[2]=="ej":
				t3.append(float(d))
				param3.append(float(l[2]))
			elif sys.argv[2]=="zj":
				t3.append(float(d))
				param3.append(float(l[3]))
			elif sys.argv[2]=="bj":
				t3.append(float(d))
				param3.append(float(l[4]))
			elif sys.argv[2]=="epsc":
				t3.append(float(d))
				param3.append(float(l[5]))

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
plt.plot(t1, param1, c=color, label="mcu "+sys.argv[3])
plt.plot(t2, param2, c=color, ls='--', label="mcu "+sys.argv[4])
plt.plot(t3, param3, c=color, ls=':', label="mcu "+sys.argv[5])
plt.legend(loc='upper left')
plt.xlabel("time [s]")
plt.ylabel(sys.argv[2])
plt.xlim([-1, 3])
#plt.axis([0, 6, 0, 20])
plt.show()
