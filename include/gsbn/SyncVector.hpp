#ifndef __GSBN_SYNC_VECTOR_HPP__
#define __GSBN_SYNC_VECTOR_HPP__

#include <cstdlib>

#include "gsbn/Common.hpp"
#include "gsbn/Fp16.hpp"
#include "gsbn/Fix16.hpp"

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
 * \class SyncVector
 * \bref SyncVector class manage the data stored in both CPU and GPU memory. It's
 * a class which wraps std::vector, thurst::host_vector and thrust::device_vector.
 */
template <typename Dtype>
class SyncVector {
public:
	
	/**
	* \enum status_t
	* \bref The status of SyncVector.
	*/
	enum status_t{
		/** Uninitialized vector.
		 * Both the CPU and GPU vector are empty.
		 */
		UNINITIALIZED,
		/** Up-to-date information stored in CPU vector.
		 * The GPU vector is either out-of-date or empty.
		 */
		CPU_VECTOR,
		/** Up-to-date information stored in GPU vector.
		 * The CPU vector is either out-of-date or empty.
		 */
		GPU_VECTOR,
		/** Both CPU and GPU vectors contain the same up-to-date inforamtion
		 */
		SYN_VECTOR
	};
	
	SyncVector();
  ~SyncVector();

	CONST_HOST_VECTOR(Dtype, *) cpu_vector();
	HOST_VECTOR(Dtype, *) mutable_cpu_vector();
	const Dtype* cpu_data(int i=0);
	Dtype* mutable_cpu_data(int i=0);
	CONST_DEVICE_VECTOR(Dtype, *) gpu_vector();
	DEVICE_VECTOR(Dtype, *) mutable_gpu_vector();
	const Dtype* gpu_data(int i=0);
	Dtype* mutable_gpu_data(int i=0);

	
	/**
	 * \fn status()
	 * \bref Get the memory type.
	 * \return The status defined in SyncVector::status_t
	 */
	status_t status(){return _status;};
	void set_ld(int l);
	int ld();
	
	int size();
	void resize(size_t s, Dtype val=0);
	void push_back(Dtype val);
	
	VectorStateI8 state_i8();
	VectorStateI16 state_i16();
	VectorStateI32 state_i32();
	VectorStateI64 state_i64();
	VectorStateF16 state_f16();
	VectorStateF32 state_f32();
	VectorStateF64 state_f64();

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
