#include "gsbn/SyncVector.hpp"

namespace gsbn{

#ifndef CPU_ONLY
template<typename Dtype>
void SyncVector<Dtype>::resize(size_t s, Dtype val){
        switch(_status){
        case GPU_VECTOR:
                _gpu_vector.resize(s, val);
		break;
        case UNINITIALIZED:
        case CPU_VECTOR:
        case SYN_VECTOR:
	default:
                _cpu_vector.resize(s, val);
		_status = CPU_VECTOR;
		break;
        }
}

template<typename Dtype>
void SyncVector<Dtype>::push_back(Dtype val){
        switch(_status){
        case GPU_VECTOR:
		_gpu_vector.push_back(val);
		break;
        case UNINITIALIZED:
        case CPU_VECTOR:
        case SYN_VECTOR:
	default:
		_cpu_vector.push_back(val);
		_status = CPU_VECTOR;
		break;
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