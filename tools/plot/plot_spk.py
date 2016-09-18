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
mcu=[]
count = np.zeros(100)

for d in subdirs:
	filename = os.path.join(sys.argv[1], d, "spk_short.txt");
	print(filename)
	with open(filename, "r") as f:
		lines = f.read().splitlines()
	for k,l in enumerate(lines):
		t.append(float(d))
		mcu.append(int(l))
		count[int(l)]+=1;

count_short = []
for m in mcu:
	count_short.append(count[m])

plt.scatter(t, mcu)
plt.xlabel("time [s]")
plt.ylabel("MCU index")
plt.xlim([-1, 3])
plt.ylim([-10, 110])
#plt.barh(mcu, count_short)
plt.show()
