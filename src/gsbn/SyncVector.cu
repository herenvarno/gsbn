#include "gsbn/SyncVector.hpp"

namespace gsbn{

#ifndef CPU_ONLY
template<typename Dtype>
void SyncVector<Dtype>::resize(size_t s, Dtype val){
        switch(_status){
        case GPU_VECTOR:
                return _gpu_vector.resize(s, val);
        case UNINITIALIZED:
        case CPU_VECTOR:
        case SYN_VECTOR:
                return _cpu_vector.resize(s, val);
        }
}

template<typename Dtype>
void SyncVector<Dtype>::push_back(Dtype val){
        switch(_status){
        case GPU_VECTOR:
                return _gpu_vector.push_back(val);
        case UNINITIALIZED:
        case CPU_VECTOR:
        case SYN_VECTOR:
                return _cpu_vector.push_back(val);
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
