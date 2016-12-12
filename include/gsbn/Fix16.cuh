#include "gsbn/Fix16.hpp"

namespace gsbn{

#ifndef CPU_ONLY
#pragma once

__device__ inline fp32 fix16_to_fp32_gpu(const fix16 in, uint32_t frac_bit) {
	return float(in)/float(1 << frac_bit);
}

__device__ inline fix16 fp32_to_fix16_gpu(const fp32 in, uint32_t frac_bit) {
	if(in>=(1 << 15-frac_bit)){
		return 0x7fff;
	}else if(in<=-(1 << (15-frac_bit))){
		return 0x8000;
	}
	return int16_t(in*(1 << frac_bit));
}
    
__device__ inline void atomic_add_fp32_to_fix16_gpu(fix16* address, float value, uint32_t frac_bit){
	uint32_t *src_addr;
	src_addr = (uint32_t *)((uint8_t*)address - ((size_t)address & 2));
	
	uint32_t old_val = atomicExch(src_addr, 0);
	uint32_t new_val = old_val;
	
	fix16 *old_h = ((fix16*)(&old_val))+1;
	fix16 *old_l = (fix16*)(&old_val);
	fix16 *new_h = ((fix16*)(&new_val))+1;
	fix16 *new_l = (fix16*)(&new_val);
	fix16 *old_ptr;
	fix16 *new_ptr;
	
	if((size_t)address & 2){
		old_ptr=old_h;
		new_ptr=new_h;
	}else{
		old_ptr=old_l;
		new_ptr=new_l;
	}
	
	*new_ptr = fp32_to_fix16_gpu(fix16_to_fp32_gpu(*old_ptr, frac_bit)+value, frac_bit);
	
	while ((old_val = atomicExch(src_addr, new_val))!=0){
		new_val = atomicExch(src_addr, 0);
		*new_l = *old_l+*new_l;
		*new_h = *old_h+*new_h;
	}
}

#endif
}
