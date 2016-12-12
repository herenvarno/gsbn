#!/usr/bin/env python

################################################################################
# This script generate random stimuli for 10x10 network.
#
# The weight mask data are:
#
# 0000000000
# 0000011111
#
################################################################################

import os
import sys
import re
import math
import random

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
pattern_num = 50

rd = gsbn_pb2.StimRawData()

rd.data_rows = 2 * pattern_num;
rd.data_cols = hcu_num;
rd.mask_rows = 2;
rd.mask_cols = hcu_num;

patterns = []

while(len(patterns)<pattern_num):
	p = []
	for j in range(mcu_num):
		p.append(random.randint(0, mcu_num-1))
	
	flag0=True
	for p0 in patterns:
		flag = True
		for idx, val in enumerate(p0):
			if(val != p[idx]):
				flag=False
				break;
		if flag==True :
			flag0=False
			break
	if flag0 == True:
		patterns.append(p)

for p in patterns:
	for v in p:
		rd.data.append(v)

for i in range(hcu_num):
	rd.mask.append(0);

for p in patterns:
	for i,v in enumerate(p):
			if i<mcu_num-1 :
				rd.data.append(v)
			else:
				rd.data.append(10000)

for i in range(hcu_num):
	if i<mcu_num-1:
		rd.mask.append(0)
	else:
		rd.mask.append(1)

with open(filename, "wb+") as f:
	f.write(rd.SerializeToString())


