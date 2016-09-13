#ifndef __GSBN_MEM_BLOCK_HPP__
#define __GSBN_MEM_BLOCK_HPP__

#include <cstdlib>

#include "gsbn/Common.hpp"

namespace gsbn {

/**
 * \class MemBlock
 * \bref MemBock class manage the data stored in both CPU and GPU memory. It
 * automatically synchronize the data when needed.
 */
class MemBlock {
public:
	
	/**
	 * \enum type_t
	 * \bref The type of memory block.
	 */
	enum type_t{
		/** Uninitialized memory block.
		 * Both the CPU and GPU memory haven't been allocated.
		*/
		UNINITIALIZED,
		/** Up-to-date information stored in CPU memory.
		 * The GPU memory is either out-of-date or hasn't been allocated.
		 */
		CPU_MEM_BLOCK,
		/** Up-to-date information stored in GPU memory.
		 * The CPU memory is either out-of-date or hasn't been allocated.
		 */
		GPU_MEM_BLOCK,
		/** Both CPU and GPU memory has the same up-to-date inforamtion
		*/
		SYN_MEM_BLOCK
	};
	
	/**
	 * \fn MemBlock()
	 * \bref A simple constructor of class MemBlock.
	 */
	MemBlock() :
		_ptr_cpu(NULL),
		_ptr_gpu(NULL),
		_size(0),
		_type(UNINITIALIZED),
		_use_cuda(false),
		_gpu_device(-1){};
	
	/**
	 * \fn MemBlock(size_t size)
	 * \bref A simple constructor of class MemBlock. with specified size.
	 * \param size The size of memory block.
	 */
	explicit MemBlock(size_t size) :
		_ptr_cpu(NULL),
		_ptr_gpu(NULL),
		_size(size),
		_type(UNINITIALIZED),
		_use_cuda(false),
		_gpu_device(-1){};
	
	/**
	 * \fn ~MemBlock()
	 * \bref The destructor of MemBlock.
	 */
  ~MemBlock();
	
	/**
	 * \fn cpu_data()
	 * \bref Get pointer of the CPU memory block. The information is garanteed to
	 * be up-to-date. The memory is read-only.
	 * \return The pointer to the memory.
	 */
	const void* cpu_data();
	/**
	 * \fn gpu_data()
	 * \bref Get pointer of the GPU memory block. The information is garanteed to
	 * be up-to-date. The memory is read-only.
	 * \return The pointer to the memory.
	 */
	const void* gpu_data();
	/**
	 * \fn mutable_cpu_data()
	 * \bref Get pointer of the CPU memory block. The information is garanteed to
	 * be up-to-date. The memory type will be set to CPU_MEM_BLOCK.
	 * \return The pointer to the memory.
	 */
	void* mutable_cpu_data();
	/**
	 * \fn mutable_gpu_data()
	 * \bref Get pointer of the GPU memory block. The information is garanteed to
	 * be up-to-date. The memory type will be set to GPU_MEM_BLOCK.
	 * \return The pointer to the memory.
	 */
	void* mutable_gpu_data();
	
	/**
	 * \fn const type_t type()
	 * \bref Get the memory type.
	 */
	const type_t type();
	/**
	 * \fn const type_t type()
	 * \bref Get the memory size.
	 */
  const size_t size();
  
  /*
   * STATIC FUNCTIONS
   */
  
  /**
	 * \fn void malloc_cpu(void** ptr, size_t size)
	 * \bref CPU version of "malloc".
	 * \param ptr The pointer of newly allocated memory, it's return value.
	 * \param size The size of required memory block.
	 * \param use_cuda The flag which shows how does the CPU memory has been allocated.
	 */
  inline static void malloc_cpu(void** ptr, size_t size, bool *use_cuda) {
		#ifndef CPU_ONLY
  	if (mode() == GPU) {
    	CUDA_CHECK(cudaMallocHost(ptr, size));
    	*use_cuda = true;
    	return;
  	}
		#endif
		*ptr = malloc(size);
		*use_cuda = false;
		CHECK(*ptr) << "host allocation of size " << size << " failed";
	}

  /**
	 * \fn void free_cpu(void* ptr)
	 * \bref CPU version of "free".
	 * \param ptr The pointer of memory.
	 * \param use_cuda The flag which shows how does the CPU memory has been allocated.
	 */
	inline static void free_cpu(void* ptr, bool use_cuda) {
		#ifndef CPU_ONLY
		if (use_cuda) {
			CUDA_CHECK(cudaFreeHost(ptr));
			return;
		}
		#endif
		free(ptr);
	}

  /**
	 * \fn void memset_cpu(void* ptr, size_t size)
	 * \bref CPU version of "memset".
	 * \param ptr The pointer of memory.
	 * \param size The size memory block.
	 */
	inline static void memset_cpu(void* ptr, int value, size_t size) {
		memset(ptr, value, size);
	}
	
	/**
	 * \fn void malloc_gpu(void** ptr, size_t size)
	 * \bref GPU version of "malloc".
	 * \param ptr The pointer of newly allocated memory, it's return value.
	 * \param size The size of required memory block.
	 */
	inline static void malloc_gpu(void** ptr, size_t size, int *gpu_device) {
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDevice(gpu_device));
		CUDA_CHECK(cudaMalloc(ptr, size));
		return;
		#endif
		__NOT_IMPLEMENTED__
	}

	/**
	 * \fn void free_gpu(void* ptr)
	 * \bref GPU version of "free".
	 * \param ptr The pointer of memory.
	 */
	inline static void free_gpu(void* ptr, int gpu_device) {
		#ifndef CPU_ONLY
		if (ptr) {
		  int initial_device;
		  cudaGetDevice(&initial_device);
		  if (gpu_device != -1) {
		    CUDA_CHECK(cudaSetDevice(gpu_device));
		  }
		  CUDA_CHECK(cudaFree(ptr));
		  cudaSetDevice(initial_device);
		}
		return;
		#endif
		__NOT_IMPLEMENTED__
	}

	/**
	 * \fn void memset_gpu(void* ptr, int value, size_t size)
	 * \bref GPU version of "memset".
	 * \param ptr The pointer of memory.
	 * \param size The size memory block.
	 */
	inline static void memset_gpu(void* ptr, int value, size_t size) {
		#ifndef CPU_ONLY
		cudaMemset	(ptr, value, size);
		return;
		#endif
		__NOT_IMPLEMENTED__
	}

	/**
	 * \fn void memcpy_gpu_to_cpu(void *ptr_to, void *ptr_from, size_t size)
	 * \bref Memory copy.
	 * \param ptr_to The pointer of destination memory.
	 * \param ptr_from The pointer of source memory.
	 * \param size The size memory block.
	 */
	inline static void memcpy_gpu_to_cpu(void *ptr_to, const void *ptr_from, size_t size) {
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaMemcpy(ptr_to, ptr_from, size, cudaMemcpyDeviceToHost));
		return;
		#endif
		__NOT_IMPLEMENTED__
	}
	
	/**
	 * \fn void memcpy_cpu_to_gpu(void *ptr_to, void *ptr_from, size_t size)
	 * \bref Memory copy.
	 * \param ptr_to The pointer of destination memory.
	 * \param ptr_from The pointer of source memory.
	 * \param size The size memory block.
	 */
	inline static void memcpy_cpu_to_gpu(void *ptr_to, const void *ptr_from, size_t size) {
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaMemcpy(ptr_to, ptr_from, size, cudaMemcpyHostToDevice));
		return;
		#endif
		__NOT_IMPLEMENTED__
	}

	/**
	 * \fn void memcpy_cpu_to_cpu(void *ptr_to, void *ptr_from, size_t size)
	 * \bref Memory copy.
	 * \param ptr_to The pointer of destination memory.
	 * \param ptr_from The pointer of source memory.
	 * \param size The size memory block.
	 */
	inline static void memcpy_cpu_to_cpu(void *ptr_to, const void *ptr_from, size_t size) {
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaMemcpy(ptr_to, ptr_from, size, cudaMemcpyHostToHost));
		return;
		#endif
		memcpy(ptr_to, ptr_from, size);
	}

	/**
	 * \fn void memcpy_gpu_to_gpu(void *ptr_to, void *ptr_from, size_t size)
	 * \bref Memory copy.
	 * \param ptr_to The pointer of destination memory.
	 * \param ptr_from The pointer of source memory.
	 * \param size The size memory block.
	 */
	inline static void memcpy_gpu_to_gpu(void *ptr_to, const void *ptr_from, size_t size) {
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaMemcpy(ptr_to, ptr_from, size, cudaMemcpyDeviceToDevice));
		return;
		#endif
		__NOT_IMPLEMENTED__
	}
  

private:
	
	void to_cpu();
	void to_gpu();
	void* _ptr_cpu;
	void* _ptr_gpu;
	size_t _size;
	type_t _type;
	bool _use_cuda;
	int _gpu_device;
};


}

#endif  //__GSBN_MEM_BLOCK_HPP__
