import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

if len(sys.argv) < 5:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <network description file> <snapshot file> <projection id> [pj|ej|zj|epsc|bj]")
	exit(-1)

network = sys.argv[1]
snapshot = sys.argv[2]
projection = int(sys.argv[3])
parameter = sys.argv[4]

# Read the network
solver_param = gsbn_pb2.SolverParam()
try:
	f = open(sys.argv[1], "r")
	text_format.Parse(f.read(), solver_param)
	f.close()
except IOError:
	print(sys.argv[1] + ": Could not open file.")
	exit(-1)

net_param = solver_param.net_param
if projection >= len(net_param.proj_param) or projection < 0 :
	print("Error Argument: projection id wrong!")
	exit(-1)

pop_param_list = net_param.pop_param
pop_param_size = len(pop_param_list)

proj_param = net_param.proj_param[projection]
src_pop = proj_param.src_pop
dest_pop = proj_param.dest_pop
if src_pop > pop_param_size or dest_pop > pop_param_size :
	print("Error Argument: network description file is wrong!")
	exit(-1)

proj_in_pop=0
for i in range(len(net_param.proj_param)):
	if net_param.proj_param[i].dest_pop==dest_pop and i!=projection:
		proj_in_pop += 1
	if i==projection:
		break;

for i in range(pop_param_size):
	if i==src_pop:
		src_pop_dim_hcu = pop_param_list[i].hcu_num;
		src_pop_dim_mcu = pop_param_list[i].mcu_num;
	if i==dest_pop:
		dest_pop_dim_hcu = pop_param_list[i].hcu_num;
		dest_pop_dim_mcu = pop_param_list[i].mcu_num;
		dest_pop_slot = pop_param_list[i].slot_num;

dest_pop_dim_conn = src_pop_dim_hcu*src_pop_dim_mcu
if(dest_pop_slot < dest_pop_dim_conn):
	dest_pop_dim_conn = dest_pop_slot

if src_pop_dim_hcu<0 or src_pop_dim_mcu<0 or dest_pop_dim_hcu<0 or dest_pop_dim_mcu<0 or dest_pop_dim_conn<0:
	print("Error Argument: network description file is wrong!")
	exit(-1)

# READ SNAPSHOT
solver_state = gsbn_pb2.SolverState()
try:
	f = open(sys.argv[2], "rb")
	solver_state.ParseFromString(f.read())
	f.close()
except IOError:
	print(sys.argv[1] + ": Could not open snapshot file.")
	exit(-1)

timestamp = solver_state.timestamp

mat = np.zeros([1, dest_pop_dim_hcu*dest_pop_dim_mcu])

if parameter=="pj" or parameter=="ej" or parameter=="zj":
	vector_state_f_list = solver_state.vector_state_f
	for i in range(len(vector_state_f_list)):
		vector_state_f = vector_state_f_list[i]
		if vector_state_f.name==parameter+"_"+str(projection):
			data = vector_state_f.data
			for j in range(len(data)):
				mat[0][j]=data[j]

if parameter=="epsc" or parameter=="bj":
	vector_state_f_list = solver_state.vector_state_f
	for i in range(len(vector_state_f_list)):
		vector_state_f = vector_state_f_list[i]
		if vector_state_f.name==parameter+"_"+str(projection):
			data = vector_state_f.data
			for j in range(len(data)):
				mat[0][j]=data[j+proj_in_pop*dest_pop_dim_hcu*dest_pop_dim_mcu]

fig, ax = plt.subplots()
cax = ax.imshow(mat,interpolation='nearest', cmap=cm.seismic)
ax.set_title("projection_"+str(projection)+"::"+parameter + " @ " + str(round(timestamp, 3))+"s")
ax.set_xlabel("mcu index (j)")
# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax)
plt.show()

