#!/usr/bin/env python

################################################################################
# This script generate stimuli for 10x10 network.
#
# The stimuli patterns are:
# 
# 0 0 0 0 0 0 0 0 0 0
# 1 1 1 1 1 1 1 1 1 1
# 2 2 2 2 2 2 2 2 2 2
# 3 3 3 3 3 3 3 3 3 3
# 4 4 4 4 4 4 4 4 4 4
# 5 5 5 5 5 5 5 5 5 5
# 6 6 6 6 6 6 6 6 6 6
# 7 7 7 7 7 7 7 7 7 7
# 8 8 8 8 8 8 8 8 8 8
# 9 9 9 9 9 9 9 9 9 9
# 0 0 0 0 0 0 0 0 0 X
# 1 1 1 1 1 1 1 1 1 X
# 2 2 2 2 2 2 2 2 2 X
# 3 3 3 3 3 3 3 3 3 X
# 4 4 4 4 4 4 4 4 4 X
# 5 5 5 5 5 5 5 5 5 X
# 6 6 6 6 6 6 6 6 6 X
# 7 7 7 7 7 7 7 7 7 X
# 8 8 8 8 8 8 8 8 8 X
# 9 9 9 9 9 9 9 9 9 X
# 0 0 0 0 0 X X X X X
# 1 1 1 1 1 X X X X X
# 2 2 2 2 2 X X X X X
# 3 3 3 3 3 X X X X X
# 4 4 4 4 4 X X X X X
# 5 5 5 5 5 X X X X X
# 6 6 6 6 6 X X X X X
# 7 7 7 7 7 X X X X X
# 8 8 8 8 8 X X X X X
# 9 9 9 9 9 X X X X X
# X X X X X X X X X X
#
# The weight mask data are:
#
# 0000000000
# 0000000001
# 0000011111
# 1111111111
#
################################################################################

import os
import sys
import re
import math
from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

if len(sys.argv) < 2:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <output file name>")
	exit(-1)

filename = sys.argv[1]

eps = 0.001
hcu_num = 10
mcu_num = 10

rd = gsbn_pb2.StimRawData()

rd.data_rows = 3 * mcu_num + 1;
rd.data_cols = hcu_num;
rd.mask_rows = 4;
rd.mask_cols = hcu_num;

for i in range(mcu_num):
	for j in range(hcu_num):
		rd.data.append(i);

for i in range(mcu_num):
	for j in range(hcu_num):
		if j!=(hcu_num-1):
			rd.data.append(i);
		else:
			rd.data.append(0x7fffffff);

for i in range(mcu_num):
	for j in range(hcu_num):
		if j<(hcu_num/2):
			rd.data.append(i);
		else:
			rd.data.append(0x7fffffff);

for i in range(1):
	for j in range(hcu_num):
		rd.data.append(0x7fffffff);

for i in range(hcu_num):
	rd.mask.append(0)

for i in range(hcu_num):
	if i!=(hcu_num-1):
		rd.mask.append(0)
	else:
		rd.mask.append(1)

for i in range(hcu_num):
	if i<(hcu_num/2):
		rd.mask.append(0)
	else:
		rd.mask.append(1)

for i in range(hcu_num):
		rd.mask.append(1)

with open(filename, "wb+") as f:
	f.write(rd.SerializeToString())


