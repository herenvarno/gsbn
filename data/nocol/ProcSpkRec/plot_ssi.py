import os
import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../build")
import gsbn_pb2

# command: ./plot_trace_t.py <description> <snapshot dir> <projection id> <trace name> <i> <j>

filename = "zj2_pop_0.csv"

lines = ""
with open(filename, "r") as f:
	lines = f.read().splitlines()

t=[]
v_zj=[]
for l in lines:
	var_list = l.split(",")
	if(len(var_list)<2):
		continue
		
	count = len(var_list)-1
	
	for i in range(count):
		t.append(int(var_list[0]))
		v_zj.append(float(var_list[1+i]))

filename = "zj2_nocol_pop_0.csv"

lines = ""
with open(filename, "r") as f:
	lines = f.read().splitlines()

v_zj_nocol=[]
for l in lines:
	var_list = l.split(",")
	if(len(var_list)<2):
		continue
		
	count = len(var_list)-1
	
	for i in range(count):
		v_zj_nocol.append(float(var_list[1+i]))
		
filename = "ssi_pop_0.csv"

lines = ""
with open(filename, "r") as f:
	lines = f.read().splitlines()

v_ssi=[]
for l in lines:
	var_list = l.split(",")
	if(len(var_list)<2):
		continue
		
	count = len(var_list)-1
	
	for i in range(count):
		v_ssi.append(float(var_list[1+i]))
		
filename = "ssj_pop_0.csv"

lines = ""
with open(filename, "r") as f:
	lines = f.read().splitlines()

v_ssj=[]
for l in lines:
	var_list = l.split(",")
	if(len(var_list)<2):
		continue
		
	count = len(var_list)-1
	
	for i in range(count):
		v_ssj.append(float(var_list[1+i]))

	
	
plt.plot(t, v_zj, t, v_zj_nocol,t,v_ssi,t,v_ssj)
plt.show()
