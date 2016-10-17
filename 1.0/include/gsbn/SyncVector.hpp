#ifndef __GSBN_SYNC_VECTOR_HPP__
#define __GSBN_SYNC_VECTOR_HPP__

#include <cstdlib>

#include "gsbn/Common.hpp"

#ifndef CPU_ONLY
#define HOST_VECTOR(Dtype, V) thrust::host_vector<Dtype> V
#define DEVICE_VECTOR(Dtype, V) thrust::device_vector<Dtype> V
#define CONST_HOST_VECTOR(Dtype, V) const thrust::host_vector<Dtype> V
#define CONST_DEVICE_VECTOR(Dtype, V) const thrust::device_vector<Dtype> V
#define HOST_VECTOR_ITERATOR(Dtype, IT) thrust::host_vector<Dtype>::iterator IT
#define DEVICE_VECTOR_ITERATOR(Dtype, IT) thrust::device_vector<Dtype>::iterator IT
#define HOST_VECTOR_CONST_ITERATOR(Dtype, IT) thrust::host_vector<Dtype>::const_iterator IT
#define DEVICE_VECTOR_CONST_ITERATOR(Dtype, IT) thrust::device_vector<Dtype>::const_iterator IT
#else
#define HOST_VECTOR(Dtype, V) std::vector<Dtype> V
#define DEVICE_VECTOR(Dtype, V) std::vector<Dtype> V
#define CONST_HOST_VECTOR(Dtype, V) const std::vector<Dtype> V
#define CONST_DEVICE_VECTOR(Dtype, V) const std::vector<Dtype> V
#define HOST_VECTOR_ITERATOR(Dtype, IT) std::vector<Dtype>::iterator IT
#define DEVICE_VECTOR_ITERATOR(Dtype, IT) std::vector<Dtype>::iterator IT
#define HOST_VECTOR_CONST_ITERATOR(Dtype, IT) std::vector<Dtype>::const_iterator IT
#define DEVICE_VECTOR_CONST_ITERATOR(Dtype, IT) std::vector<Dtype>::const_iterator IT
#endif

namespace gsbn {

/**
 * \class MemBlock
 * \bref MemBock class manage the data stored in both CPU and GPU memory. It
 * automatically synchronize the data when needed.
 */
template <typename Dtype>
class SyncVector {
public:
	
	/**
	 * \enum type_t
	 * \bref The type of memory block.
	 */
	enum status_t{
		/** Uninitialized memory block.
		 * Both the CPU and GPU memory haven't been allocated.
		*/
		UNINITIALIZED,
		/** Up-to-date information stored in CPU memory.
		 * The GPU memory is either out-of-date or hasn't been allocated.
		 */
		CPU_VECTOR,
		/** Up-to-date information stored in GPU memory.
		 * The CPU memory is either out-of-date or hasn't been allocated.
		 */
		GPU_VECTOR,
		/** Both CPU and GPU memory has the same up-to-date inforamtion
		*/
		SYN_VECTOR
	};
	
	SyncVector();
  ~SyncVector();

	const HOST_VECTOR(Dtype, *) cpu_vector();
	HOST_VECTOR(Dtype, *) mutable_cpu_vector();
	const Dtype* cpu_data(int i=0);
	Dtype* mutable_cpu_data(int i=0);
	const DEVICE_VECTOR(Dtype, *) gpu_vector();
	DEVICE_VECTOR(Dtype, *) mutable_gpu_vector();
	const Dtype* gpu_data(int i=0);
	Dtype* mutable_gpu_data(int i=0);
	
	/**
	 * \fn status()
	 * \bref Get the memory type.
	 * \return The status defined in SyncVector::status_t
	 */
	const status_t status(){return _status;};
  void set_ld(int l);
	const int ld();
	
	VectorStateI state_i();
	VectorStateF state_f();
	VectorStateD state_d();

private:
	
	void to_cpu();
	void to_gpu();
	
	int _ld;
	HOST_VECTOR(Dtype, _cpu_vector);
	DEVICE_VECTOR(Dtype, _gpu_vector);
	
	status_t _status;
};


	
}

#endif  //__GSBN_MEM_BLOCK_HPP__
