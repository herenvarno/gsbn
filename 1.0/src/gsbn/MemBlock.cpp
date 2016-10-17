#include "gsbn/MemBlock.hpp"

namespace gsbn {

MemBlock::~MemBlock() {
	if (_ptr_cpu) {
		free_cpu(_ptr_cpu, _use_cuda);
	}
	
	//FIXME: Program hasn't compile with cuda, all code execute on CPU.
  if (_ptr_gpu) {
		free_gpu(_ptr_gpu, _gpu_device);	
	}
}

inline void MemBlock::to_cpu() {
	switch (_type) {
	case UNINITIALIZED:
		malloc_cpu(&_ptr_cpu, _size, &_use_cuda);
    memset_cpu(_ptr_cpu, 0, _size);
    _type = CPU_MEM_BLOCK;
    break;
	case GPU_MEM_BLOCK:
    if (_ptr_cpu == NULL) {
			malloc_cpu(&_ptr_cpu, _size, &_use_cuda);
    }
    memcpy_gpu_to_cpu(_ptr_cpu, _ptr_gpu, _size);
    _type = SYN_MEM_BLOCK;
    break;
	case CPU_MEM_BLOCK:
	case SYN_MEM_BLOCK:
		break;
	}
}

inline void MemBlock::to_gpu() {
	switch (_type) {
	case UNINITIALIZED:
		malloc_gpu(&_ptr_gpu, _size, &_gpu_device);
		memset_gpu(_ptr_gpu, 0, _size);
    _type = GPU_MEM_BLOCK;
    break;
	case CPU_MEM_BLOCK:
    if (_ptr_gpu == NULL) {
			malloc_gpu(&_ptr_gpu, _size, &_gpu_device);
    }
    memcpy_cpu_to_gpu(_ptr_gpu, _ptr_cpu, _size);
    _type = SYN_MEM_BLOCK;
    break;
	case GPU_MEM_BLOCK:
	case SYN_MEM_BLOCK:
		break;
	}
}

const void* MemBlock::cpu_data() {
	to_cpu();
	return (const void*)_ptr_cpu;
}
const void* MemBlock::gpu_data() {
	to_gpu();
	return (const void*)_ptr_gpu;
}

void* MemBlock::mutable_cpu_data() {
	to_cpu();
	_type = CPU_MEM_BLOCK;
  return _ptr_cpu;
}
void* MemBlock::mutable_gpu_data() {
  to_gpu();
  _type = GPU_MEM_BLOCK;
  return _ptr_gpu;
}


const MemBlock::type_t MemBlock::type() {
	return _type;
}

const size_t MemBlock::size() {
	return _size;
}

}
