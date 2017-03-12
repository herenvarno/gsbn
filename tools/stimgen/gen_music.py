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

music = [
	[6,0,0,0,6,0,0,0,7,0,0,0,8,0,0,0,8,0,0,0,7,0,0,0,6,0,0,0,5,0,0,0],
	[9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]

mcu_num = 14
hcu_num = len(music)
cycle = len(music[0])

conn_pop_hcu_num = 100

rd = gsbn_pb2.StimRawData()

rd.data_rows = cycle+1;
rd.data_cols = hcu_num+conn_pop_hcu_num;
rd.mask_rows = 2;
rd.mask_cols = hcu_num+conn_pop_hcu_num;

for i in range(cycle):
	for j in range(hcu_num):
		rd.data.append(music[j][i])
	for j in range(conn_pop_hcu_num):
		rd.data.append(0x7fffffff);
for j in range(hcu_num+conn_pop_hcu_num):
	rd.data.append(0x7fffffff);

for x in range(rd.data_rows):
	print(rd.data[x*rd.data_cols:(x+1)*rd.data_cols])
		
for i in range(hcu_num):
	rd.mask.append(0)
for i in range(conn_pop_hcu_num):
	rd.mask.append(1)
for i in range(rd.mask_cols):
	rd.mask.append(1)

with open(filename, "wb+") as f:
	f.write(rd.SerializeToString())

