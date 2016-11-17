import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <spike record file> <population id>")
	exit(-1)

spike_file = sys.argv[1]
population = int(sys.argv[2])
t=[]
mcu=[]

with open(spike_file, "r") as f:
	lines = f.read().splitlines()

for l in lines:
	var_list = l.split(",")
	if(len(var_list)<2):
		continue
	if(int(var_list[1])!=population):
		continue
		
	mcu_count = len(var_list)-2
	
	for i in range(mcu_count):
		t.append(float(var_list[0]))
		mcu.append(int(var_list[2+i]))


plt.scatter(t, mcu)
plt.xlabel("time [s]")
plt.ylabel("MCU index")
plt.show()
