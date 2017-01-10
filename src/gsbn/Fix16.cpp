#include "gsbn/Fix16.hpp"

namespace gsbn{


float fix16_to_fp32(const fix16 in, int frac_bit) {
	CHECK_GE(frac_bit,0);
	return float(in)/float(1 << frac_bit);
}

fix16 fp32_to_fix16(const float in, int frac_bit) {
	CHECK_GE(frac_bit,0);
	if(frac_bit <=15){
		if(in>=(1 << 15-frac_bit)){
			return 0x7fff;
		}else if(in<=-(1 << (15-frac_bit))){
			return 0x8000;
		}
	}else{
		if(in * (1 << frac_bit-15) >= 1){
			return 0x7fff;
		}else if(in * (1 << frac_bit-15) <= -1){
			return 0x8000;
		}
	}
	return int16_t(in*(1 << frac_bit));
}

float fix32_to_fp32(const fix32 in, int frac_bit) {
	CHECK_GE(frac_bit,0);
	return float(in)/float(1 << frac_bit);
}

fix32 fp32_to_fix32(const float in, int frac_bit) {
	CHECK_GE(frac_bit,0);
	if(frac_bit <=31){
		if(in>=(1 << 31-frac_bit)){
			return 0x7fffffff;
		}else if(in<=-(1 << (31-frac_bit))){
			return 0x80000000;
		}
	}else{
		if(in * (1 << frac_bit-31) >= 1){
			return 0x7fffffff;
		}else if(in * (1 << frac_bit-31) <= -1){
			return 0x80000000;
		}
	}
	return int32_t(in*(1 << frac_bit));
}
}
