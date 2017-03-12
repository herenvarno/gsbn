#!/usr/bin/env python

################################################################################
# This script generate random stimuli for single-population BCPNN network.
################################################################################

import os
import sys
import re
import math
import random

from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

if len(sys.argv) < 6:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <hcu num> <mcu num> <pattern num> <silent rate> <output file name>")
	exit(-1)

filename = sys.argv[5]

hcu_num = int(sys.argv[1])
mcu_num = int(sys.argv[2])
pattern_num = int(sys.argv[3])

silent_rate = float(sys.argv[4])

rd = gsbn_pb2.StimRawData()

rd.data_rows = 2 * pattern_num;
rd.data_cols = hcu_num;
rd.mask_rows = 2;
rd.mask_cols = hcu_num;

patterns = []

while(len(patterns)<pattern_num):
	p = []
	for j in range(hcu_num):
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
		if i<hcu_num*(1-silent_rate) :
			rd.data.append(v)
		else:
			rd.data.append(0x7fffffff)

for i in range(hcu_num):
	if i<hcu_num*(1-silent_rate):
		rd.mask.append(0)
	else:
		rd.mask.append(1)

#with open(filename+".list", "w+") as f:
#	print("Patterns:", file=f)
#	for i in range(pattern_num):
#		print(rd.data[i*hcu_num:(i+1)*hcu_num], file=f)
#	print("", file=f)
#	print("Masks:", file=f)
#	print(rd.mask[0:hcu_num], file=f)
#	print(rd.mask[hcu_num:2*hcu_num], file=f)

with open(filename, "wb+") as f:
	f.write(rd.SerializeToString())


