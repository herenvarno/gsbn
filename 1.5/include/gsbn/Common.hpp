#ifndef __GSBN_COMMON_HPP__
#define __GSBN_COMMON_HPP__

#include "util/easylogging++.h"
#include "gsbn.pb.h"
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef CPU_ONLY

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/fill.h>

#endif

using namespace std;

namespace gsbn{

/** \mainpage GSBN -- GPU version of Spiking-based BCPNN
 *
 * \section Introduction
 *
 * The BCPNN
 *
 * \section Structure
 * The GSBN provide a simulation environment called Solver. In the Solver, there
 * is a Gen (Generator), a Net (Network), a Rec (Recorder) as well as a Database.
 * The Database mantains all the data which defines the current state.
 * The Gen updates the timestamp, controls the simulation step and chooses the
 * correct stimuli. The Net update the state of BCPNN according to its internal
 * logic. The Rec generate the snapshot of Database and save it to a binary file.
 *
 * \section Data 
 * In Database of a Solver, the data is organized as Table. While, the Database
 * is created based on a text description file or restored from a binary snapshot.
 * 
 * The text description file is a protobuf format text file, which defines the
 * network. The text description is usually provided by the user to create
 * a new Solver.
 *
 * The binary snapshot file or state file is usually generated by Rec from previous
 * training. With the snapshot file, a complete Solver can be recovered including
 * the correct timestamp.
 *
 * Another important file is the stimili file which should be provided along with
 * text description file in order to create a new Solver. (Attention: This part may be changed
 * due to new mechanism of stimulation).
 */

enum _mode_t{
	CPU,
	GPU
};

mode_t mode();
void set_mode(mode_t m);

enum _source_t{
	NEW,
	COPY
};
_source_t source();
void set_source(_source_t s);

#define __NOT_IMPLEMENTED__ LOG(FATAL) << "Function hasn't been implemented";

#ifdef CPU_ONLY

#define __NO_GPU__ LOG(FATAL) << "Cannot use GPU in CPU-only mode: check mode.";

#else

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << gsbn::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << gsbn::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int GSBN_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GSBN_GET_BLOCKS(const int N) {
  return (N + GSBN_CUDA_NUM_THREADS - 1) / GSBN_CUDA_NUM_THREADS;
}
inline int GSBN_GET_THREADS(const int N) {
	if(N>=GSBN_CUDA_NUM_THREADS){
		return GSBN_CUDA_NUM_THREADS;
	}else{
		return N;
	}
}

void SetDevice(const int device_id);
void DeviceQuery();
bool CheckDevice(const int device_id);
int FindDevice(const int start_id);

#endif

}

#endif //__GSBN_COMMON_HPP__
