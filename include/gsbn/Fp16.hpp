#ifndef __GSBN_FP16_HPP__
#define __GSBN_FP16_HPP__

#include "gsbn/Common.hpp"

namespace gsbn{

#define fp16 uint16_t
#define fp32 float
#define fp64 double

fp32 fp16_to_fp32(const fp16 in);
fp16 fp32_to_fp16(const fp32 in);

#ifndef CPU_ONLY
__device__ inline fp32 fp16_to_fp32_gpu(const fp16 in);
__device__ inline fp16 fp32_to_fp16_gpu(const fp32 in);
__device__ inline void atomic_add_fp32_to_fp16_gpu(fp16* address, fp32 value, fp16* max_address);
__device__ inline void atomic_add_fp16_to_fp16_gpu(fp16* address, fp16 value, fp16* max_address);
#endif

}

#endif
