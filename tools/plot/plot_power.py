import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

if(len(sys.argv) < 2):
	print("too few arguments");
	exit(0)

LOG_FILE = sys.argv[1]

print(LOG_FILE)

lines = []
with open(LOG_FILE) as f:
	lines = f.read().split('\n')

time_list = []
power_list = []
for i,line in enumerate(lines):
	m = re.match(r"P\d+, \d+, ([\d\.]+)", line)
	if m :
		time_list.append((i*2)/1000)
		power_list.append(float(m.groups()[0]))

plt.plot(time_list, power_list)
plt.show()
