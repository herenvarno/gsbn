#!/usr/bin/env python

################################################################################
# This script generate stimuli for 10x10 network.
#
# The stimuli patterns are:
# 
# 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000
# 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000
# 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000
# 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000
# 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000
# 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000
# 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000
# 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100
# 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010
# 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001
# 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 0000000000
# 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0100000000 0000000000
# 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0010000000 0000000000
# 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0001000000 0000000000
# 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000100000 0000000000
# 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000010000 0000000000
# 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000001000 0000000000
# 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000100 0000000000
# 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000010 0000000000
# 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000001 0000000000
#
# The weight mask data are:
#
# 0000000000
# 0000000001
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

rd.data_rows = 2 * mcu_num;
rd.data_cols = hcu_num * mcu_num;
rd.mask_rows = 2;
rd.mask_cols = hcu_num;

for i in range(mcu_num):
	for j in range(hcu_num):
		for k in range(mcu_num):
			if i==k:
				rd.data.append(math.log(1+eps));
			else:
				rd.data.append(math.log(0+eps));

for i in range(hcu_num):
	rd.mask.append(0);

for i in range(mcu_num):
	for j in range(hcu_num):
		for k in range(mcu_num):
			if i==k and j!=(hcu_num-1):
				rd.data.append(math.log(1+eps));
			else:
				rd.data.append(math.log(0+eps));

for i in range(hcu_num):
	if i!=(hcu_num-1):
		rd.mask.append(0)
	else:
		rd.mask.append(1)

with open(filename, "wb+") as f:
	f.write(rd.SerializeToString())


