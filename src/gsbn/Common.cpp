#include "gsbn/Common.hpp"

namespace gsbn{


static mode_t _mode=CPU;

mode_t mode(){
	return _mode;
}

void set_mode(mode_t m){
	_mode = m;
}

static _source_t _source=NEW;

_source_t source(){
	return _source;
}

void set_source(_source_t s){
	_source = s;
}

#ifndef CPU_ONLY
static cublasHandle_t _cublas_handle;

inline cublasHandle_t cublas_handle(){
	return _cublas_handle;
}
#endif


#ifdef CPU_ONLY  // CPU-only Caffe.

void SetDevice(const int device_id) {
  __NO_GPU__;
}

void DeviceQuery() {
  __NO_GPU__;
}

bool CheckDevice(const int device_id) {
  __NO_GPU__;
}

int FindDevice(const int start_id) {
  __NO_GPU__;
}

#else


void SetDevice(const int device_id) {
	__NOT_IMPLEMENTED__
}

void DeviceQuery() {
	__NOT_IMPLEMENTED__
}

bool CheckDevice(const int device_id) {
	__NOT_IMPLEMENTED__
}

int FindDevice(const int start_id) {
	__NOT_IMPLEMENTED__
}





const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#endif


void Common_init(int* argc, char ***argv){
	
	GlobalVar glv;
	int rank_global;
	int num_rank_global;
	int rank_local;
	char processor_name[512];
	int processor_name_size;
	
	MPI_Init(argc, argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);
	MPI_Comm_size(MPI_COMM_WORLD, &num_rank_global);
	MPI_Get_processor_name(processor_name, &processor_name_size);
	processor_name[processor_name_size]='\0';
	
	MPI_Win win_node_names;
	MPI_Win win_local_ranks;
	vector<char> shared_node_names;
	vector<int> shared_local_ranks;
	if(rank_global == 0){
		shared_node_names.resize(num_rank_global * 512);
		shared_local_ranks.resize(num_rank_global);
		MPI_Win_create(&shared_node_names[0], shared_node_names.size(), sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &win_node_names);
		MPI_Win_create(&shared_local_ranks[0], shared_local_ranks.size(), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_local_ranks);
	}else{
		MPI_Win_create(MPI_BOTTOM, 0, sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &win_node_names);
		MPI_Win_create(MPI_BOTTOM, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_local_ranks);
	}
	
	
	MPI_Win_fence(0, win_node_names);
	MPI_Put(&processor_name[0], processor_name_size, MPI_CHAR, 0, rank_global * 512, processor_name_size, MPI_CHAR, win_node_names);
	MPI_Win_fence(0, win_node_names);
	
	if(rank_global == 0){
		vector<int> local_ranks;
		vector<string> node_names;
		for(int i=0; i<num_rank_global; i++){
			string n(&shared_node_names[i*512]);
			int j=0;
			for(j=0; j<node_names.size(); j++){
				if(n==node_names[j]){
					shared_local_ranks[i] = local_ranks[j];
					local_ranks[j]++;
					break;
				}
			}
			if(j>=node_names.size()){
				shared_local_ranks[i] = 0;
				local_ranks.push_back(1);
				node_names.push_back(n);
			}
		}
	}
	
	MPI_Win_fence(0, win_local_ranks);
	MPI_Get(&rank_local, 1, MPI_INT, 0, rank_global, 1, MPI_INT, win_local_ranks);
	MPI_Win_fence(0, win_local_ranks);
	
	glv.puti("rank", rank_global);
	glv.puti("num-rank", num_rank_global);
	glv.puti("rank-local", rank_local);
	
	#ifndef CPU_ONLY
	int num_gpu_local=0;
	cudaGetDeviceCount(&num_gpu_local);
	glv.puti("num-gpu", num_gpu_local);
	if(rank_local<num_gpu_local){
		cudaSetDevice(rank_local);
	}
	#endif
	
	MPI_Win_free(&win_node_names);
	MPI_Win_free(&win_local_ranks);
}

}
