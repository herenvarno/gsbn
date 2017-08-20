#ifndef __GSBN_SYNC_VECTOR_HPP__
#define __GSBN_SYNC_VECTOR_HPP__

#include <cstdlib>
#include "gsbn/Common.hpp"
#include "gsbn/Conv.hpp"

#ifdef CPU_ONLY
/*
 * CPU MODE
 */

#define HOST_VECTOR(Dtype, V) std::vector<Dtype> V
#define CONST_HOST_VECTOR(Dtype, V) const std::vector<Dtype> V
#define HOST_VECTOR_ITERATOR(Dtype, IT) std::vector<Dtype>::iterator IT
#define HOST_VECTOR_CONST_ITERATOR(Dtype, IT) std::vector<Dtype>::const_iterator IT

#else

#define HOST_VECTOR(Dtype, V) thrust::host_vector<Dtype> V
#define DEVICE_VECTOR(Dtype, V) thrust::device_vector<Dtype> V
#define CONST_HOST_VECTOR(Dtype, V) const thrust::host_vector<Dtype> V
#define CONST_DEVICE_VECTOR(Dtype, V) const thrust::device_vector<Dtype> V
#define HOST_VECTOR_ITERATOR(Dtype, IT) thrust::host_vector<Dtype>::iterator IT
#define DEVICE_VECTOR_ITERATOR(Dtype, IT) thrust::device_vector<Dtype>::iterator IT
#define HOST_VECTOR_CONST_ITERATOR(Dtype, IT) thrust::host_vector<Dtype>::const_iterator IT
#define DEVICE_VECTOR_CONST_ITERATOR(Dtype, IT) thrust::device_vector<Dtype>::const_iterator IT

#endif

namespace gsbn {

/**
 * \class SyncVector
 * \brief SyncVector class manage the data stored in both CPU and GPU memory. It's
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

#ifdef CPU_ONLY

template <typename Dtype>
class SyncVector {
public:
	
	enum status_t{
		OUT_OF_DATE,
		UP_TO_DATE
	};
	
	enum type_t{
		HOST
	};
	
	struct vec_t{
		type_t _t;
		status_t _s;
		std::vector<Dtype> _v_host;
	};
	
	SyncVector();
	~SyncVector();
	
	void register_device(int id, type_t type);
	
	const std::vector<Dtype>* host_vector(int id);
	std::vector<Dtype>* mutable_host_vector(int id);
	const Dtype* host_data(int id, int i=0);
	Dtype* mutable_host_data(int id, int i=0);
	
	inline void set_ld(int l);
	int ld();
	
	int size();
	void resize(size_t s, Dtype val=0, bool lazy=true);
	void push_back(Dtype val, bool lazy=true);
	
	VectorStateI8 state_i8();
	VectorStateI16 state_i16();
	VectorStateI32 state_i32();
	VectorStateI64 state_i64();
	VectorStateF16 state_f16();
	VectorStateF32 state_f32();
	VectorStateF64 state_f64();
	
	/*
	 * Deprecated
	 */
	const std::vector<Dtype>* cpu_vector();
	std::vector<Dtype>* mutable_cpu_vector();
	const Dtype* cpu_data(int i=0);
	Dtype* mutable_cpu_data(int i=0);

private:
	void to_device(int id);
	
	int _ld;
	map<int, vec_t> _vectors;
};

#else

template <typename Dtype>
class SyncVector {
public:
	
	enum status_t{
		OUT_OF_DATE,
		UP_TO_DATE
	};
	
	enum type_t{
		HOST,
		DEVICE
	};
	
	struct vec_t{
		type_t _t;
		status_t _s;
		thrust::host_vector<Dtype> _v_host;
		thrust::device_vector<Dtype> _v_device;
	};
	
	SyncVector();
	~SyncVector();
	
	void register_device(int id, type_t type);
	
	const thrust::host_vector<Dtype>* host_vector(int id);
	thrust::host_vector<Dtype>* mutable_host_vector(int id);
	const Dtype* host_data(int id, int i=0);
	Dtype* mutable_host_data(int id, int i=0);
	const thrust::device_vector<Dtype>* device_vector(int id);
	thrust::device_vector<Dtype>* mutable_device_vector(int id);
	const Dtype* device_data(int id, int i=0);
	Dtype* mutable_device_data(int id, int i=0);
	
	inline void set_ld(int l);
	int ld();
	
	int size();
	void resize(size_t s, Dtype val=0, bool lazy=true);
	void push_back(Dtype val, bool lazy=true);
	
	VectorStateI8 state_i8();
	VectorStateI16 state_i16();
	VectorStateI32 state_i32();
	VectorStateI64 state_i64();
	VectorStateF16 state_f16();
	VectorStateF32 state_f32();
	VectorStateF64 state_f64();
	
	/*
	 * Deprecated
	 */
	const thrust::host_vector<Dtype>* cpu_vector();
	thrust::host_vector<Dtype>* mutable_cpu_vector();
	const Dtype* cpu_data(int i=0);
	Dtype* mutable_cpu_data(int i=0);
	const thrust::device_vector<Dtype>* gpu_vector();
	thrust::device_vector<Dtype>* mutable_gpu_vector();
	const Dtype* gpu_data(int i=0);
	Dtype* mutable_gpu_data(int i=0);

	void print_state();
	
private:
	void to_device(int id);
	void assign_vec(vec_t& a, vec_t& b);
	
	int _ld;
	map<int, vec_t> _vectors;
};
#endif
}

#endif  //__GSBN_SYNC_VECTOR_HPP__
