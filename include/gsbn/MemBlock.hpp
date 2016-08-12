#ifndef __GSBN_MEM_BLOCK_HPP__
#define __GSBN_MEM_BLOCK_HPP__

#include <cstdlib>

#include "gsbn/Common.hpp"

namespace ghcu {


inline void malloc_cpu(void** ptr, size_t size) {
	*ptr = malloc(size);
	CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void free_cpu(void* ptr) {
  free(ptr);
}

inline void memset_cpu(void* ptr, size_t size) {
	memset(ptr, 0, size);
}



//FIXME: Program hasn't compile with cuda, all code execute on CPU.
inline void malloc_gpu(void** ptr, size_t size) {
	*ptr = malloc(size);
	CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void free_gpu(void* ptr) {
  free(ptr);
}

inline void memset_gpu(void* ptr, size_t size) {
	memset(ptr, 0, size);
}

inline void memcpy_gpu_to_cpu(void *ptr_to, void *ptr_from, size_t size) {
	memcpy(ptr_to, ptr_from, size);
}

inline void memcpy_cpu_to_gpu(void *ptr_to, void *ptr_from, size_t size) {
	memcpy(ptr_to, ptr_from, size);
}

inline void memcpy_cpu_to_cpu(void *ptr_to, void *ptr_from, size_t size) {
	memcpy(ptr_to, ptr_from, size);
}

inline void memcpy_gpu_to_gpu(void *ptr_to, void *ptr_from, size_t size) {
	memcpy(ptr_to, ptr_from, size);
}


class MemBlock {
public:
	
	enum type_t{
		UNINITIALIZED,
		CPU_MEM_BLOCK,
		GPU_MEM_BLOCK,
		SYN_MEM_BLOCK
	};
	
	MemBlock() :
		_ptr_cpu(NULL),
		_ptr_gpu(NULL),
		_size(0),
		_type(UNINITIALIZED){};
	explicit MemBlock(size_t size) :
		_ptr_cpu(NULL),
		_ptr_gpu(NULL),
		_size(size),
		_type(UNINITIALIZED){};
  ~MemBlock();
	
	const void* cpu_data();
	const void* gpu_data();
	void* mutable_cpu_data();
	void* mutable_gpu_data();
//	void set_cpu_data(void *data, size_t size);
//	void set_gpu_data(void *data, size_t size);
	
  type_t type() const { return _type; }
  size_t size() { return _size; }

private:
	
	void to_cpu();
	void to_gpu();
	void* _ptr_cpu;
	void* _ptr_gpu;
	size_t _size;
	type_t _type;

};


}

#endif  //__GSBN_MEM_BLOCK_HPP__
