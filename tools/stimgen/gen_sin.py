#!/usr/bin/env python

import os
import sys
import re
import math
import random
import matplotlib.pyplot as plt
import numpy as np

from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

if len(sys.argv) < 1:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <output file name>")
	exit(-1)

filename = sys.argv[1]

patterns = []
masks = []

DIM_HCU = 10
DIM_MCU = 10

rd = gsbn_pb2.StimRawData()

rd.data_rows = 5;
rd.data_cols = DIM_HCU;
rd.mask_rows = 3;
rd.mask_cols = DIM_HCU;

p = [0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff]
patterns.append(p)
p = [0,1,2,3,4,5,6,7,8,9]
patterns.append(p)
p = [9,8,7,6,5,4,3,2,1,0]
patterns.append(p)
p = [0,1,2,3,4,5,6,7,8,0x7fffffff]
patterns.append(p)
p = [9,8,7,6,5,4,3,2,1,0x7fffffff]
patterns.append(p)

m = [0,0,0,0,0,0,0,0,0,0]
masks.append(m)
m = [1,1,1,1,1,1,1,1,1,1]
masks.append(m)
m = [0,0,0,0,0,0,0,0,0,1]
masks.append(m)

for p in patterns:
	for v in p:
		rd.data.append(v)

for p in masks:
	for v in p:
		rd.mask.append(v)

with open(filename, "wb+") as f:
	f.write(rd.SerializeToString())
