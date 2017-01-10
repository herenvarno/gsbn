import os
import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

# command: ./plot_trace_t.py <description> <snapshot dir> <projection id> <trace name> <i> <j>

if len(sys.argv) < 7:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <network description file> <snapshot file> <projection id> [pij|eij|zi2|zj2|tij|wij]")
	exit(-1)
	
network = sys.argv[1]
snapshot = sys.argv[2]
projection = int(sys.argv[3])
parameter = sys.argv[4]
cord_i = int(sys.argv[5])
cord_j = int(sys.argv[6])

# Read the network
solver_param = gsbn_pb2.SolverParam()
try:
	f = open(network, "r")
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
trace=[]

os.chdir(snapshot)
for f in glob.glob("SolverState*.bin"):
	print(f)
	solver_state = gsbn_pb2.SolverState()
	try:
		f = open(f, "rb")
		solver_state.ParseFromString(f.read())
		f.close()
	except IOError:
		print(sys.argv[1] + ": Could not open snapshot file.")
		exit(-1)
	
	timestamp = solver_state.timestamp
	ii = np.zeros([dest_pop_dim_hcu*dest_pop_dim_conn])
	vector_state_i32_list = solver_state.vector_state_i32
	for i in range(len(vector_state_i32_list)):
		vector_state_i32 = vector_state_i32_list[i]
		if vector_state_i32.name=="ii_"+str(projection):
			data = vector_state_i32.data
			for j in range(len(data)):
				ii[j]=int(data[j])
		
	if parameter=="pi" or parameter=="ei" or parameter=="zi":
		vector_state_f32_list = solver_state.vector_state_f32
		for i in range(len(vector_state_f32_list)):
			vector_state_f32 = vector_state_f32_list[i]
			if vector_state_f32.name==parameter+"_"+str(projection):
				data = vector_state_f32.data
				for j in range(len(data)):
					y=ii[j]
					x=j//dest_pop_dim_conn
					if y==cord_i and x==cord_j and y>=0:
						trace.append([timestamp, data[j]])
	
	if parameter=="pij" or parameter=="eij" or parameter=="zi2" or parameter=="zj2":
		vector_state_f32_list = solver_state.vector_state_f32
		for i in range(len(vector_state_f32_list)):
			vector_state_f32 = vector_state_f32_list[i]
			if vector_state_f32.name==parameter+"_"+str(projection):
				data = vector_state_f32.data
				for j in range(len(data)):
					h=j//dest_pop_dim_mcu
					w=j%dest_pop_dim_mcu
					y=ii[h];
					x=h//dest_pop_dim_conn*dest_pop_dim_mcu+w
					if y==cord_i and x==cord_j and y>=0:
						trace.append([timestamp, data[j]])

print(trace)
time=[]
value=[]
trace.sort(key=lambda x: x[0])
for v in trace:
	time.append(v[0])
	value.append(v[1])
print(time)
print(value)
plt.plot(time, value)
plt.show()
