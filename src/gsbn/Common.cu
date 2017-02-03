#include "gsbn/Common.hpp"

namespace gsbn{

#ifndef CPU_ONLY

static cublasHandle_t _cublas_handle;

cublasHandle_t cublas_handle(){
	return _cublas_handle;
}

void common_init(){
CUBLAS_CHECK(cublasCreate(&_cublas_handle));
}

#endif

}
