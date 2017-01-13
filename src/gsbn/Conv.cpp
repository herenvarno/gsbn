#include "gsbn/Conv.hpp"

namespace gsbn{

fp32 fp16_to_fp32(const fp16 in){
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
fp16 fp32_to_fp16(const fp32 in){
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

fp32 fx32_to_fp32(const fx32 in, uint32_t frac_bit){
	return float(in)/float(1 << frac_bit);
}
fx32 fp32_to_fx32(const fp32 in, uint32_t frac_bit){
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
fp32 fx16_to_fp32(const fx16 in, uint32_t frac_bit){
	return fp32(in)/fp32(1 << frac_bit);
}
fx16 fp32_to_fx16(const fp32 in, uint32_t frac_bit){
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
fp32 fx8_to_fp32(const fx8 in, uint32_t frac_bit){
	return fp32(in)/fp32(1 << frac_bit);
}
fx8 fp32_to_fx8_gpu(const fp32 in, uint32_t frac_bit){
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

fp32 ufx32_to_fp32(const ufx32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__; return 0;}
ufx32 fp32_to_ufx32(const fp32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__; return 0;}
fp32 ufx16_to_fp32(const ufx16 in, uint32_t frac_bit){__NOT_IMPLEMENTED__; return 0;}
ufx16 fp32_to_ufx16(const fp32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__; return 0;}
fp32 ufx8_to_fp32(const ufx8 in, uint32_t frac_bit){__NOT_IMPLEMENTED__; return 0;}
ufx8 fp32_to_ufx8(const fp32 in, uint32_t frac_bit){__NOT_IMPLEMENTED__; return 0;}

}
