#include "gsbn/Fix16.hpp"

namespace gsbn{
float fix16_to_fp32(const fix16 in) {
	return float(in)/1024.0;
}
fix16 fp32_to_fix16(const float in) {
	if(in>=32){
		return 0x7fff;
	}else if(in<=-32){
		return 0x1000;
	}
	return int16_t(in*1024);
}

float fix16_15_to_fp32(const fix16 in) {
	return float(in)/32768.0;
}
fix16 fp32_to_fix16_15(const float in) {
	if(in>=1){
		return 0x7fff;
	}else if(in<=-1){
		return 0x1000;
	}
	return int16_t(in*32768.0);
}
}
