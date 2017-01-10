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
 * 
 * The SyncVector stores data inside vector containers. The vector containers can
 * be implemented by std::vector, thrust::host_vector or thrust::device_vector.
 * The std::vector belongs to the std library while the other two belong to cuda::thrust
 * library. std::vector and thrust::host_vector store data inside the main memory,
 * thrust::device_vector on the other hand, stores data in GPU global memory.
 *
 * The SyncVector has two vector containers whoes data is supposed to be considered
 * synchronized at all time. The two vector container are _cpu_vector (std::vector
 * or thrust::host_vector) and _gpu_vector (thrust::device_vector). In fact, the
 * data stored inside both vectors are not always the same, because the cost would
 * be too high if we update one vector everytime the other vector is changed. Therefore,
 * class SyncVector is designed properly to deal with data synchronization when
 * needed.
 * 
 * There are two sets of API for accessing the data of SyncVector. cpu_vector(),
 * gpu_vector(), cpu_data() and gpu_data() return read-only pointers which avoid
 * modification; while mutable_cpu_vector(), * mutable_gpu_vector(), mutable_cpu_data()
 * and mutable_gpu_data() return normal pointers which enables modification of the
 * data inside SyncVector.
 * 
 * In order to make SyncVector more flexiable, the data stored inside the class
 * can either be 1D array or 2D matrix. The shape of data depends on private parameter
 * leading-dimension (_ld) which represents the number of point in each row.
 * if _ld=1, in each row, there is only 1 point, so it's an array, while _ld>1, it's a 2D matrix.
 * _ld can be set and read by API function set_ld() and ld().
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

	/**
	 * \fn cpu_vector()
	 * \bref Get the read-only pointer of the cpu vector which stores its data in
	 * main memory.
	 * \return The read-only pointer of the cpu vector.
	 */
	CONST_HOST_VECTOR(Dtype, *) cpu_vector();
	/**
	 * \fn mutable_cpu_vector()
	 * \bref Get the pointer of the cpu vector which stores its data in
	 * main memory.
	 * \return The pointer of the cpu vector.
	 */
	HOST_VECTOR(Dtype, *) mutable_cpu_vector();
	/**
	 * \fn cpu_data()
	 * \bref Get the read-only pointer of row i of the data stored in cpu vector.
	 * \param i The row index if considering the data block as a 2D matrix.
	 * \return The read-only pointer of the data.
	 */
	const Dtype* cpu_data(int i=0);
	/**
	 * \fn mutable_cpu_data()
	 * \bref Get the pointer of row i of the data stored in cpu vector.
	 * \param i The row index if considering the data block as a 2D matrix.
	 * \return The pointer of the data.
	 */
	Dtype* mutable_cpu_data(int i=0);
	/**
	 * \fn gpu_vector()
	 * \bref Get the read-only pointer of the gpu vector which stores its data in
	 * GPU global memory.
	 * \return The read-only pointer of the gpu vector.
	 */
	CONST_DEVICE_VECTOR(Dtype, *) gpu_vector();
	/**
	 * \fn mutable_gpu_vector()
	 * \bref Get the pointer of the gpu vector which stores its data in
	 * GPU global.
	 * \return The pointer of the gpu vector.
	 */
	DEVICE_VECTOR(Dtype, *) mutable_gpu_vector();
	/**
	 * \fn gpu_data()
	 * \bref Get the read-only pointer of row i of the data stored in gpu vector.
	 * \param i The row index if considering the data block as a 2D matrix.
	 * \return The read-only pointer of the data.
	 */
	const Dtype* gpu_data(int i=0);
	/**
	 * \fn mutable_gpu_data()
	 * \bref Get the pointer of row i of the data stored in gpu vector.
	 * \param i The row index if considering the data block as a 2D matrix.
	 * \return The pointer of the data.
	 */
	Dtype* mutable_gpu_data(int i=0);

	
	/**
	 * \fn status()
	 * \bref Get the memory type.
	 * \return The status defined in SyncVector::status_t
	 */
	inline status_t status(){return _status;};
	/**
	 * \fn set_ld()
	 * \bref Set the leading dimension.
	 * \param l The leading dimension, l>=0.
	 */
	inline void set_ld(int l);
	/**
	 * \fn ld()
	 * \bref Get the leading dimension.
	 * \return The leading dimension.
	 */
	inline int ld();
	
	/**
	 * \fn size()
	 * \bref Similar to std::vector<Dtype>::size(). It returns the size of the
	 * vector which contains the most up-to-date data.
	 * \return The size.
	 */
	int size();
	/**
	 * \fn resize()
	 * \bref Similar to std::vector::resize(). It reshapes the vector which contains
	 * the most up-to-data data.
	 * \param s The new size.
	 * \param val The default value used to fill the newly allocated space.
	 * \warning if the new size is smaller than the old size, the data which stored beyond
	 * the new size will be discarded.
	 */
	void resize(size_t s, Dtype val=0);
	/**
	 * \fn push_back()
	 * \bref Similar to std::vector<Dtype>::push_back(). It appends a new value to
	 * the vector which contains the most up-to-data data.
	 * \param val The value to append.
	 */
	void push_back(Dtype val);
	
	/**
	 * \fn state_i8()
	 * \bref Pack the data to a protobuf message.
	 * \return The packed message.
	 * \warning The function always considers each value as int8_t type. If the data
	 * is in other types, please use the proper function instead.
	 */
	VectorStateI8 state_i8();
	/**
	 * \fn state_i16()
	 * \bref Pack the data to a protobuf message.
	 * \return The packed message.
	 * \warning The function always considers each value as int16_t type. If the real data
	 * is in other types, please use the proper function instead.
	 */
	VectorStateI16 state_i16();
	/**
	 * \fn state_i32()
	 * \bref Pack the data to a protobuf message.
	 * \return The packed message.
	 * \warning The function always considers each value as int32_t type. If the real data
	 * is in other types, please use the proper function instead.
	 */
	VectorStateI32 state_i32();
	/**
	 * \fn state_i64()
	 * \bref Pack the data to a protobuf message.
	 * \return The packed message.
	 * \warning The function always considers each value as int64_t type. If the real data
	 * is in other types, please use the proper function instead.
	 */
	VectorStateI64 state_i64();
	/**
	 * \fn state_f16()
	 * \bref Pack the data to a protobuf message.
	 * \return The packed message.
	 * \warning The function always considers each value as half float type. Half
	 * float type is the macro of uint16_t. If the real data
	 * is in other types, please use the proper function instead.
	 */
	VectorStateF16 state_f16();
	/**
	 * \fn state_f32()
	 * \bref Pack the data to a protobuf message.
	 * \return The packed message.
	 * \warning The function always considers each value as 32-bit float type. If the real data
	 * is in other types, please use the proper function instead.
	 */
	VectorStateF32 state_f32();
	/**
	 * \fn state_f64()
	 * \bref Pack the data to a protobuf message.
	 * \return The packed message.
	 * \warning The function always considers each value as 64-bit double float type. If the real data
	 * is in other types, please use the proper function instead.
	 */
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

#endif  //__GSBN_SYNC_VECTOR_HPP__
