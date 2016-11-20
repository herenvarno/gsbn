#!/usr/bin/env python

################################################################################
# Generate sin function training data
################################################################################

import os
import sys
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

if len(sys.argv) < 2:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <output file name>")
	exit(-1)

filename = sys.argv[1]

x = np.arange(10)
x = x[::-1]
print(x)
x_train = []
y_train = []
x_test = []
y_test = []
for i in x:
	y = int(math.sin((i/10)*math.pi*2)*5)+5
	#y = 9-i

	x_train.append(i)
	y_train.append(y)
	x_test.append(i)
	y_test.append(y)


#plt.subplot(2, 1, 1)
#plt.scatter(x_train, y_train, c='white')
#plt.subplot(2, 1, 2)
#plt.scatter(x_test, y_test, c='red')
#
#plt.show()


eps = 0.001
hcu_num = 10
mcu_num = 10

rd = gsbn_pb2.StimRawData()

rd.data_rows = len(x_train)+len(x_test);
rd.data_cols = hcu_num * mcu_num;
rd.mask_rows = 2;
rd.mask_cols = hcu_num;

print(rd.data_rows, rd.data_cols, rd.mask_rows, rd.mask_cols)

for i in range(len(x_train)):
	for k in range(hcu_num-1):
		for j in range(mcu_num):
			if j==x_train[i]:
				rd.data.append(math.log(1+eps));
			else:
				rd.data.append(math.log(0+eps));
	for j in range(mcu_num):
		if j==y_train[i]:
			rd.data.append(math.log(1+eps));
		else:
			rd.data.append(math.log(0+eps));

for i in range(hcu_num):
	rd.mask.append(0)

for i in range(len(x_test)):
	for k in range(hcu_num-1):
		for j in range(mcu_num):
			if j==x_test[i]:
				rd.data.append(math.log(1+eps));
			else:
				rd.data.append(math.log(0+eps));
	for j in range(mcu_num):
		if j==y_test[i]:
			rd.data.append(math.log(0+eps));
		else:
			rd.data.append(math.log(0+eps));

for i in range(hcu_num):
	if i!=hcu_num-1:
		rd.mask.append(0)
	elif i==hcu_num-1:
		rd.mask.append(1)
	else:
		rd.mask.append(1)

print(len(rd.data), len(rd.mask))

with open(filename, "wb+") as f:
	f.write(rd.SerializeToString())


