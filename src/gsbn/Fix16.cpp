#include "gsbn/Fix16.hpp"

namespace gsbn{


float fix16_to_fp32(const fix16 in, int frac_bit) {
	CHECK_GE(frac_bit,0);
	return float(in)/float(1 << frac_bit);
}

fix16 fp32_to_fix16(const float in, int frac_bit) {
	CHECK_GE(frac_bit,0);
	if(in>=(1 << 15-frac_bit)){
		return 0x7fff;
	}else if(in<=-(1 << (15-frac_bit))){
		return 0x8000;
	}
	return int16_t(in*(1 << frac_bit));
}

}
