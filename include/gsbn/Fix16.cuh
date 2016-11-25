#include "gsbn/Fix16.hpp"

namespace gsbn{

#ifndef CPU_ONLY
#pragma once

__device__ inline fp32 fix16_to_fp32_gpu(const fix16 in) {
	return float(in)/1024.0;
}

__device__ inline fp16 fp32_to_fp16_gpu(const fp32 in) {
	return int16_t(in*1024);
}
    
__device__ inline void atomic_add_fix16_gpu(fix16* address, fix16 value, fix16* max_address){
	uint32_t *src_addr;
	src_addr = (uint32_t *)((uint8_t*)address - ((size_t)address & 2));
	
	uint32_t old_val = atomicExch(src_addr, 0);
	uint32_t new_val = old_val;
	
	fp16 *old_h = ((fp16*)(&old_val))+1;
	fp16 *old_l = (fp16*)(&old_val);
	fp16 *new_h = ((fp16*)(&new_val))+1;
	fp16 *new_l = (fp16*)(&new_val);
	fp16 *old_ptr;
	fp16 *new_ptr;
	
	if((size_t)address & 2){
		old_ptr=old_h;
		new_ptr=new_h;
	}else{
		old_ptr=old_l;
		new_ptr=new_l;
	}
	
	*new_ptr = *old_ptr+value;
	
	while ((old_val = atomicExch(src_addr, new_val))!=0){
		new_val = atomicExch(src_addr, 0);
		*new_l = *old_l+*new_l;
		*new_h = *old_h+*new_h;
	}
}

#endif
}
