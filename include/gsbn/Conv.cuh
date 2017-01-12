#include "gsbn/Conv.hpp"

namespace gsbn{

#ifndef CPU_ONLY
#pragma once

__device__ inline fp32 fp16_to_fp32_gpu(const fp16 in);
__device__ inline fp16 fp32_to_fp16_gpu(const fp32 in);

__device__ inline fp32 fx32_to_fp32_gpu(const fx32 in, uint32_t frac_bit);
__device__ inline fx32 fp32_to_fx32_gpu(const fp32 in, uint32_t frac_bit);
__device__ inline fp32 fx16_to_fp32_gpu(const fx16 in, uint32_t frac_bit);
__device__ inline fx16 fp32_to_fx16_gpu(const fp32 in, uint32_t frac_bit);
__device__ inline fp32 fx8_to_fp32_gpu(const fx8 in, uint32_t frac_bit);
__device__ inline fx8 fp32_to_fx8_gpu(const fp32 in, uint32_t frac_bit);

__device__ inline fp32 ufx32_to_fp32_gpu(const ufx32 in, uint32_t frac_bit);
__device__ inline ufx32 fp32_to_ufx32_gpu(const fp32 in, uint32_t frac_bit);
__device__ inline fp32 ufx16_to_fp32_gpu(const ufx16 in, uint32_t frac_bit);
__device__ inline ufx16 fp32_to_ufx16_gpu(const fp32 in, uint32_t frac_bit);
__device__ inline fp32 ufx8_to_fp32_gpu(const ufx8 in, uint32_t frac_bit);
__device__ inline ufx8 fp32_to_ufx8_gpu(const fp32 in, uint32_t frac_bit);

__device__ inline void atomic_add_fp32_to_fp16_gpu(fp16* address, fp32 value);
__device__ inline void atomic_add_fp32_to_fx32_gpu(fx32* address, fp32 value);
__device__ inline void atomic_add_fp32_to_fx16_gpu(fx16* address, fp32 value);
__device__ inline void atomic_add_fp32_to_fx8_gpu(fx8* address, fp32 value);
__device__ inline void atomic_add_fp32_to_ufx32_gpu(ufx32* address, fp32 value);
__device__ inline void atomic_add_fp32_to_ufx16_gpu(ufx16* address, fp32 value);
__device__ inline void atomic_add_fp32_to_ufx8_gpu(ufx8* address, fp32 value);

__device__ inline fp32 fp16_to_fp32_gpu(const fp16 in){
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
__device__ inline fp16 fp32_to_fp16_gpu(const fp32 in){
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

__device__ inline fp32 fx32_to_fp32_gpu(const fx32 in, uint32_t frac_bit){
	return float(in)/float(1 << frac_bit);
}
__device__ inline fx32 fp32_to_fx32_gpu(const fp32 in, uint32_t frac_bit){
	if(frac_bit <=31){
		if(in>=(1 << (31-frac_bit))){
			return 0x7fffffff;
		}else if(in<=-(1 << (31-frac_bit))){
			return 0x80000000;
		}
	}else{
		if(in * (1 << (frac_bit-31)) >= 1){
			return 0x7fffffff;
		}else if(in * (1 << (frac_bit-31)) <= -1){
			return 0x80000000;
		}
	}
	return (fx32)(in*(1 << frac_bit));
}
__device__ inline fp32 fx16_to_fp32_gpu(const fx16 in, uint32_t frac_bit){
	return fp32(in)/fp32(1 << frac_bit);
}
__device__ inline fx16 fp32_to_fx16_gpu(const fp32 in, uint32_t frac_bit){
	if(frac_bit <=15){
		if(in>=(1 << (15-frac_bit))){
			return 0x7fff;
		}else if(in<=-(1 << (15-frac_bit))){
			return 0x8000;
		}
	}else{
		if(in * (1 << (frac_bit-15)) >= 1){
			return 0x7fff;
		}else if(in * (1 << (frac_bit-15)) <= -1){
			return 0x8000;
		}
	}
	return (fx16)(in*(1 << frac_bit));
}
__device__ inline fp32 fx8_to_fp32_gpu(const fx8 in, uint32_t frac_bit){
	return fp32(in)/fp32(1 << frac_bit);
}
__device__ inline fx8 fp32_to_fx8_gpu(const fp32 in, uint32_t frac_bit){
	if(frac_bit <=7){
		if(in>=(1 << (7-frac_bit))){
			return 0x7f;
		}else if(in<=-(1 << (7-frac_bit))){
			return 0x80;
		}
	}else{
		if(in * (1 << (frac_bit-7)) >= 1){
			return 0x7f;
		}else if(in * (1 << (frac_bit-7)) <= -1){
			return 0x80;
		}
	}
	return (fx8)(in*(1 << frac_bit));
}

__device__ inline fp32 ufx32_to_fp32_gpu(const ufx32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__;}
__device__ inline ufx32 fp32_to_ufx32_gpu(const fp32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__;}
__device__ inline fp32 ufx16_to_fp32_gpu(const ufx16 in, uint32_t frac_bit){__NOT_IMPLEMENTED__;}
__device__ inline ufx16 fp32_to_ufx16_gpu(const fp32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__;}
__device__ inline fp32 ufx8_to_fp32_gpu(const ufx8 in, uint32_t frac_bit){__NOT_IMPLEMENTED__;}
__device__ inline ufx8 fp32_to_ufx8_gpu(const fp32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__;}

__device__ inline void atomic_add_fp32_to_fp16_gpu(fp16* address, fp32 value){
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
	
	*new_ptr = fp32_to_fp16_gpu(fp16_to_fp32_gpu(*old_ptr)+value);
	
	while ((old_val = atomicExch(src_addr, new_val))!=0){
		new_val = atomicExch(src_addr, 0);
		*new_l = fp32_to_fp16_gpu(fp16_to_fp32_gpu(*old_l)+fp16_to_fp32_gpu(*new_l));
		*new_h = fp32_to_fp16_gpu(fp16_to_fp32_gpu(*old_h)+fp16_to_fp32_gpu(*new_h));
	}
}
__device__ inline void atomic_add_fp32_to_fx32_gpu(fx32* address, fp32 value){
	uint32_t *src_addr;
	src_addr = (uint32_t *)(address);
	
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
		*new_l = fp32_to_fix16_gpu(fix16_to_fp32_gpu(*old_l, frac_bit)+fix16_to_fp32_gpu(*new_l, frac_bit), frac_bit);
		*new_h = fp32_to_fix16_gpu(fix16_to_fp32_gpu(*old_h, frac_bit)+fix16_to_fp32_gpu(*new_h, frac_bit), frac_bit);
	}
}
__device__ inline void atomic_add_fp32_to_fx16_gpu(fx16* address, fp32 value){
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
		*new_l = fp32_to_fix16_gpu(fix16_to_fp32_gpu(*old_l, frac_bit)+fix16_to_fp32_gpu(*new_l, frac_bit), frac_bit);
		*new_h = fp32_to_fix16_gpu(fix16_to_fp32_gpu(*old_h, frac_bit)+fix16_to_fp32_gpu(*new_h, frac_bit), frac_bit);
	}
}
__device__ inline void atomic_add_fp32_to_fx8_gpu(fx8* address, fp32 value){__NOT_IMPLEMENTED__;}
__device__ inline void atomic_add_fp32_to_ufx32_gpu(ufx32* address, fp32 value){__NOT_IMPLEMENTED__;}
__device__ inline void atomic_add_fp32_to_ufx16_gpu(ufx16* address, fp32 value){__NOT_IMPLEMENTED__;}
__device__ inline void atomic_add_fp32_to_ufx8_gpu(ufx8* address, fp32 value){__NOT_IMPLEMENTED__;}

#endif

}

