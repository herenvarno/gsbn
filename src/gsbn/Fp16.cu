#include "gsbn/Fp16.hpp"

namespace gsbn{

#ifndef CPU_ONLY

fp32 fp16_to_fp32_gpu(const fp16 in) {
	uint32_t t1;
	uint32_t t2;
	uint32_t t3;
	fp32 out;

	t1 = in & 0x7fff;                       // Non-sign bits
	t2 = in & 0x8000;                       // Sign bit
	t3 = in & 0x7c00;                       // Exponent
        
	t1 <<= 13;                              // Align mantissa on MSB
	t2 <<= 16;                              // Shift sign bit into position

	t1 += 0x38000000;                       // Adjust bias
	t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero
	t1 |= t2;                               // Re-insert sign bit

	*((uint32_t*)(&out)) = t1;
	return out;
}

fp16 fp32_to_fp16_gpu(const fp32 in) {
	uint32_t inu = *((uint32_t*)&in);
	uint32_t t1;
	uint32_t t2;
	uint32_t t3;
	fp16 out;

	t1 = inu & 0x7fffffff;                 // Non-sign bits
	t2 = inu & 0x80000000;                 // Sign bit
	t3 = inu & 0x7f800000;                 // Exponent
        
	t1 >>= 13;                             // Align mantissa on MSB
	t2 >>= 16;                             // Shift sign bit into position

	t1 -= 0x1c000;                         // Adjust bias

	t1 = (t3 < 0x38800000) ? 0 : t1;       // Flush-to-zero
	t1 = (t3 > 0x8e000000) ? 0x7bff : t1;  // Clamp-to-max
	t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

	t1 |= t2;                              // Re-insert sign bit

	*((uint16_t*)(&out)) = t1;
	return out;
}
    
__device__ inline void atomic_add_fp32_to_fp16_gpu(fp16* address, fp32 value, fp16* max_address){
	bool last_flag = false;
	uint32_t *src_addr;
	if(address==max_address){
		src_addr = (uint32_t *)(address-1);
		last_flag = true;
	}else{
		src_addr = (uint32_t *)(address);
	}
	uint32_t old = atomicExch(src_addr, 0);
	uint32_t new = old;
	
	fp16 *old_h = (fp16*)(&old)+1;
	fp16 *old_l = (fp16*)(&old);
	fp16 *new_h = (fp16*)(&new)+1;
	fp16 *new_l = (fp16*)(&new);
	fp16 *old_ptr;
	fp16 *new_ptr;
	
	if(last_flag){
		old_ptr=old_h;
		new_ptr=new_h;
	}else{
		old_ptr=old_l;
		new_ptr=new_l;
	}
	
	*new_ptr = fp32_to_fp16(fp16_to_fp32_gpu(*old_ptr)+value);
	
	while ((old = atomicExch(src_addr, new))!=0){
		new = atomicExch(src_addr, 0);
		*new_l = fp32_to_fp16_gpu(fp16_to_fp32_gpu(*old_l)+fp16_to_fp32_gpu(*new_l));
		*new_h = fp32_to_fp16_gpu(fp16_to_fp32_gpu(*old_h)+fp16_to_fp32_gpu(*new_h));
	}
}

__device__ inline void atomic_add_fp16_to_fp16_gpu(fp16* address, fp16 value, fp16* max_address){
	bool last_flag = false;
	uint32_t *src_addr;
	if(address==max_address){
		src_addr = (uint32_t *)(address-1);
		last_flag = true;
	}else{
		src_addr = (uint32_t *)(address);
	}
	uint32_t old = atomicExch(src_addr, 0);
	uint32_t new = old;
	
	fp16 *old_h = (fp16*)(&old)+1;
	fp16 *old_l = (fp16*)(&old);
	fp16 *new_h = (fp16*)(&new)+1;
	fp16 *new_l = (fp16*)(&new);
	fp16 *old_ptr;
	fp16 *new_ptr;
	
	if(last_flag){
		old_ptr=old_h;
		new_ptr=new_h;
	}else{
		old_ptr=old_l;
		new_ptr=new_l;
	}
	
	*new_ptr = fp32_to_fp16(fp16_to_fp32_gpu(*old_ptr)+fp16_to_fp32_gpu(value));
	
	while ((old = atomicExch(src_addr, new))!=0){
		new = atomicExch(src_addr, 0);
		*new_l = fp32_to_fp16_gpu(fp16_to_fp32_gpu(*old_l)+fp16_to_fp32_gpu(*new_l));
		*new_h = fp32_to_fp16_gpu(fp16_to_fp32_gpu(*old_h)+fp16_to_fp32_gpu(*new_h));
	}
}

#endif
}
