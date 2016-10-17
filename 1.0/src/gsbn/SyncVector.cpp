#include "gsbn/SyncVector.hpp"

namespace gsbn{

#ifndef CPU_ONLY
template <typename Dtype>
SyncVector<Dtype>::SyncVector():
	_ld(1), _cpu_vector(), _gpu_vector(), _status(UNINITIALIZED){

}
#else
template <typename Dtype>
SyncVector<Dtype>::SyncVector():
	_ld(1), _cpu_vector(), _status(UNINITIALIZED){

}
#endif

template <typename Dtype>
SyncVector<Dtype>::~SyncVector(){

}

template <typename Dtype>
const HOST_VECTOR(Dtype, *) SyncVector<Dtype>::cpu_vector(){
	to_cpu();
	return (const HOST_VECTOR(Dtype, *))(&_cpu_vector);
}
template <typename Dtype>
HOST_VECTOR(Dtype, *) SyncVector<Dtype>::mutable_cpu_vector(){
	to_cpu();
	_status=CPU_VECTOR;
	return (HOST_VECTOR(Dtype, *))(&_cpu_vector);
}
template <typename Dtype>
const Dtype* SyncVector<Dtype>::cpu_data(int i){
	CHECK_GE(i, 0);
	if(_cpu_vector.empty()){
		return NULL;
	}
	to_cpu();
	if(i==0){
		return (const Dtype*)(&(_cpu_vector[0]));
	}else{
		return (const Dtype*)(&(_cpu_vector[i*_ld]));
	}
}
template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_cpu_data(int i){
	CHECK_GE(i, 0);
	if(_cpu_vector.empty()){
		return NULL;
	}
	to_cpu();
	_status=CPU_VECTOR;
	if(i==0){
		return (Dtype*)(&(_cpu_vector[0]));
	}else{
		return (Dtype*)(&(_cpu_vector[i*_ld]));
	}
}
template <typename Dtype>
const DEVICE_VECTOR(Dtype, *) SyncVector<Dtype>::gpu_vector(){
	#ifndef CPU_ONLY
	to_gpu();
	return (const DEVICE_VECTOR(Dtype, *))(&_gpu_vector);
	#else
	__NO_GPU__;
	#endif
}
template <typename Dtype>
DEVICE_VECTOR(Dtype, *) SyncVector<Dtype>::mutable_gpu_vector(){
	#ifndef CPU_ONLY
	to_gpu();
	_status=GPU_VECTOR;
	return (DEVICE_VECTOR(Dtype, *))(&_gpu_vector);
	#else
	__NO_GPU__;
	#endif
}
template <typename Dtype>
const Dtype* SyncVector<Dtype>::gpu_data(int i){
	CHECK_GE(i, 0);
	#ifndef CPU_ONLY
	if(_gpu_vector.empty()){
		return NULL;
	}
	to_gpu();
	if(i==0){
		return (const Dtype*)(&(_gpu_vector[0]));
	}else{
		return (const Dtype*)(&(_gpu_vector[i*_ld]));
	}
	#else
	__NO_GPU__;
	#endif
}
template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_gpu_data(int i){
	CHECK_GE(i, 0);
	#ifndef CPU_ONLY
	if(_gpu_vector.empty()){
		return NULL;
	}
	to_gpu();
	_status=GPU_VECTOR;
	if(i==0){
		return (Dtype*)(&(_gpu_vector[0]));
	}else{
		return (Dtype*)(&(_gpu_vector[i*_ld]));
	}
	#else
	__NO_GPU__;
	#endif
}

template <typename Dtype>
void SyncVector<Dtype>::set_ld(int l){
	CHECK_GT(l, 0);
	_ld = l;
}

template <typename Dtype>
const int SyncVector<Dtype>::ld(){
	return _ld;
}

template <typename Dtype>
VectorStateI SyncVector<Dtype>::state_i(){
	VectorStateI vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const int)((*v)[i]));
	}
	return vs;
}
template <typename Dtype>
VectorStateF SyncVector<Dtype>::state_f(){
	VectorStateF vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const float)((*v)[i]));
	}
	return vs;
}
template <typename Dtype>
VectorStateD SyncVector<Dtype>::state_d(){
	VectorStateD vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const double)((*v)[i]));
	}
	return vs;
}

////////////////////////////////////////////////////////////////////////////////
// Private functions
////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void SyncVector<Dtype>::to_cpu(){
	#ifndef CPU_ONLY
	switch(_status){
	case GPU_VECTOR:
		_cpu_data = gpu_data;
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
		_gpu_data = cpu_data;
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
