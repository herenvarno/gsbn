#include "gsbn/SyncVector.hpp"

namespace gsbn{

////////////////////////////////////////////////////////////////////////////////
// Private functions
////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void SyncVector<Dtype>::to_cpu(){
	#ifndef CPU_ONLY
	switch(_status){
	case GPU_VECTOR:
		//_cpu_vector.resize(_gpu_vector.size());
		_cpu_vector = _gpu_vector;
		_status = SYN_VECTOR;
		break;
	case UNINITIALIZED:
	case CPU_VECTOR:
	case SYN_VECTOR:
		break;
	}
	#endif
}
template <typename Dtype>
void SyncVector<Dtype>::to_gpu(){
	#ifndef CPU_ONLY
	switch(_status){
	case CPU_VECTOR:
		//_gpu_vector.resize(_cpu_vector.size());
		_gpu_vector = _cpu_vector;
		_status = SYN_VECTOR;
		break;
	case UNINITIALIZED:
	case GPU_VECTOR:
	case SYN_VECTOR:
		break;
	}
	#endif
}


template class SyncVector<int>;
template class SyncVector<float>;
template class SyncVector<double>;

}
