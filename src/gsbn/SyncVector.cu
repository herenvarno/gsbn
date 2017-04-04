#include "gsbn/SyncVector.hpp"

namespace gsbn{

#ifndef CPU_ONLY
template<typename Dtype>
void SyncVector<Dtype>::resize(size_t s, Dtype val, bool lazy){
	bool updated=false;
	for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
		if(itx->second._s==UP_TO_DATE){
			if(updated==false || lazy==false){
				switch(itx->second._t){
				case HOST:
					itx->second._v_host.resize(s, val);
					break;
				case DEVICE:
					itx->second._v_device.resize(s, val);
					break;
				}
			}else{
				itx->second._s==OUT_OF_DATE;
			}
		}
	}
}

template<typename Dtype>
void SyncVector<Dtype>::push_back(Dtype val, bool lazy){
	bool updated=false;
	for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
		if(itx->second._s==UP_TO_DATE){
			if(updated==false || lazy==false){
				switch(itx->second._t){
				case HOST:
					itx->second._v_host.push_back(val);
					break;
				case DEVICE:
					itx->second._v_device.push_back(val);
					break;
				}
			}else{
				itx->second._s==OUT_OF_DATE;
			}
		}
	}
}
#endif

template class SyncVector<int8_t>;
template class SyncVector<int16_t>;
template class SyncVector<int32_t>;
template class SyncVector<int64_t>;
template class SyncVector<fp16>;
template class SyncVector<float>;
template class SyncVector<double>;

}
