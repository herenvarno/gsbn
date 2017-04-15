import os
import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2


class AniPop:
	def __init__(self, args):
		if len(args)<4:
			print("Too few Argument!")
			exit(-1)
	
		self.pop_id = int(args[2])
		self.trace = args[3]
		
		#READ NETWORK
		network_file = args[0]
		solver_param = gsbn_pb2.SolverParam()
		try:
			f = open(network_file, "r")
			text_format.Merge(str(f.read()), solver_param)
			f.close()
		except IOError:
			print(network_file + ": Could not open file.")
			exit(-1)
		
		net_param = solver_param.net_param
		if self.pop_id >= len(net_param.pop_param) or self.pop_id < 0 :
			print("Error Argument: population id is wrong!")
			exit(-1)
		
		pop_param = net_param.pop_param[self.pop_id]
		self.dim_hcu = pop_param.hcu_num
		self.dim_mcu = pop_param.mcu_num
		
		if self.dim_hcu<0 or self.dim_mcu<0 :
			print("Error Argument: population id is wrong!")
			exit(-1)
		
		#GENERATE SNAPSHOT LIST
		snapshot_dir = args[1]
		self.snapshots = glob.glob(os.path.abspath(snapshot_dir)+"/*.bin")
		self.snapshots.sort(self.sort_callback)
		
		# DRAW FIGURE
		self.data = np.zeros([self.dim_hcu, self.dim_mcu])
		self.fig, self.ax = plt.subplots()
		self.cax = self.ax.imshow(self.data, interpolation='nearest', cmap=cm.seismic)
		self.ax.set_title("Pop_"+str(self.pop_id)+"::"+ self.trace)
		self.ax.set_xlabel("MCU index")
		self.ax.set_ylabel("HCU index")
	
	def init(self):
		
		
	def update(self, i):
		
	
	def show(self):
		plt.show()
		
	def sort_callback(self, a, b):
		m1 = re.search('_([\.\d]+).bin$', a)
		m2 = re.search('_([\.\d]+).bin$', b)
		if float(m1.group(1))>float(m2.group(1)):
			return 1
		elif float(m1.group(1))==float(m2.group(1)):
			return 0
		else:
			return -1

if __name__ == '__main__':
	ani = AniPop(sys.argv[1:])
	ani.show()


#if len(sys.argv) < 5:
#	print("Arguments wrong! Please retry with command :")
#	print("python "+os.path.realpath(__file__)+" <network description file> <snapshot dir> <population id> [act|dsup]")
#	exit(-1)




#network = sys.argv[1]
#snapshot = sys.argv[2]
#projection = int(sys.argv[3])
#parameter = sys.argv[4]

## Read the network
#solver_param = gsbn_pb2.SolverParam()
#try:
#	f = open(sys.argv[1], "r")
#	text_format.Parse(f.read(), solver_param)
#	f.close()
#except IOError:
#	print(sys.argv[1] + ": Could not open file.")
#	exit(-1)

#net_param = solver_param.net_param
#if projection >= len(net_param.proj_param) or projection < 0 :
#	print("Error Argument: projection id wrong!")
#	exit(-1)

#pop_param_list = net_param.pop_param
#pop_param_size = len(pop_param_list)

#proj_param = net_param.proj_param[projection]
#src_pop = proj_param.src_pop
#dest_pop = proj_param.dest_pop
#if src_pop > pop_param_size or dest_pop > pop_param_size :
#	print("Error Argument: network description file is wrong!")
#	exit(-1)

#proj_in_pop=0
#for i in range(len(net_param.proj_param)):
#	if net_param.proj_param[i].dest_pop==dest_pop and i!=projection:
#		proj_in_pop += 1
#	if i==projection:
#		break;

#for i in range(pop_param_size):
#	if i==src_pop:
#		src_pop_dim_hcu = pop_param_list[i].hcu_num;
#		src_pop_dim_mcu = pop_param_list[i].mcu_num;
#	if i==dest_pop:
#		dest_pop_dim_hcu = pop_param_list[i].hcu_num;
#		dest_pop_dim_mcu = pop_param_list[i].mcu_num;

#dest_pop_slot = proj_param.slot_num;
#dest_pop_dim_conn = src_pop_dim_hcu*src_pop_dim_mcu
#if(dest_pop_slot < dest_pop_dim_conn):
#	dest_pop_dim_conn = dest_pop_slot

#if src_pop_dim_hcu<0 or src_pop_dim_mcu<0 or dest_pop_dim_hcu<0 or dest_pop_dim_mcu<0 or dest_pop_dim_conn<0:
#	print("Error Argument: network description file is wrong!")
#	exit(-1)

## READ SNAPSHOT
#solver_state = gsbn_pb2.SolverState()
#try:
#	f = open(sys.argv[2], "rb")
#	solver_state.ParseFromString(f.read())
#	f.close()
#except IOError:
#	print(sys.argv[1] + ": Could not open snapshot file.")
#	exit(-1)

#timestamp = solver_state.timestamp

#mat = np.zeros([dest_pop_dim_hcu, dest_pop_dim_mcu])

#if parameter=="pj" or parameter=="ej" or parameter=="zj":
#	vector_state_f32_list = solver_state.vector_state_f32
#	for i in range(len(vector_state_f32_list)):
#		vector_state_f32 = vector_state_f32_list[i]
#		if vector_state_f32.name==parameter+"_"+str(projection):
#			data = vector_state_f32.data
#			for j in range(len(data)):
#				mat[j//dest_pop_dim_mcu][j%dest_pop_dim_mcu]=data[j]

#if parameter=="epsc" or parameter=="bj":
#	vector_state_f32_list = solver_state.vector_state_f32
#	for i in range(len(vector_state_f32_list)):
#		vector_state_f32 = vector_state_f32_list[i]
#		if vector_state_f32.name==parameter+"_"+str(projection):
#			data = vector_state_f32.data
#			for j in range(len(data)):
#				mat[j//dest_pop_dim_mcu][j%dest_pop_dim_mcu]=data[j+proj_in_pop*dest_pop_dim_hcu*dest_pop_dim_mcu]

#fig, ax = plt.subplots()
#cax = ax.imshow(mat,interpolation='nearest', cmap=cm.seismic)
#ax.set_title("projection_"+str(projection)+"::"+parameter + " @ " + str(round(timestamp, 3))+"s")
#ax.set_xlabel("mcu index (internal index in each HCU)")
#ax.set_ylabel("hcu index")
## Add colorbar, make sure to specify tick locations to match desired ticklabels
#cbar = fig.colorbar(cax)
#plt.show()

