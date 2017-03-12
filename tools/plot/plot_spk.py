import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <spike record file>")
	exit(-1)

spike_file = sys.argv[1]
t=[]
mcu=[]

with open(spike_file, "r") as f:
	lines = f.read().splitlines()

for l in lines:
	var_list = l.split(",")
	if(len(var_list)<2):
		continue
		
	mcu_count = len(var_list)-1
	
	for i in range(mcu_count):
		t.append(float(var_list[0]))
		mcu.append(int(var_list[1+i]))


plt.scatter(t, mcu)
plt.xlabel("Cycle")
plt.ylabel("MCU index")
plt.show()
