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
	to_cpu();
	if(_cpu_vector.empty()){
		return NULL;
	}
	#ifndef CPU_ONLY
	if(i==0){
                return (const Dtype*)(thrust::raw_pointer_cast(_cpu_vector.data()));
        }else{
                return (const Dtype*)(thrust::raw_pointer_cast(_cpu_vector.data()+i*_ld));
        }
	#else
	if(i==0){
		return (const Dtype*)(&(_cpu_vector[0]));
	}else{
		return (const Dtype*)(&(_cpu_vector[i*_ld]));
	}
	#endif
}
template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_cpu_data(int i){
	CHECK_GE(i, 0);
	to_cpu();
	_status=CPU_VECTOR;
	if(_cpu_vector.empty()){
		return NULL;
	}
	#ifndef CPU_ONLY
	if(i==0){
		return (Dtype*)(thrust::raw_pointer_cast(_cpu_vector.data()));
	}else{
		return (Dtype*)(thrust::raw_pointer_cast(_cpu_vector.data()));
	}
	#else
	if(i==0){
		return (Dtype*)(&(_cpu_vector[0]));
	}else{
		return (Dtype*)(&(_cpu_vector[i*_ld]));
	}
	#endif
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
	to_gpu();
	if(_gpu_vector.empty()){
		return NULL;
	}
	if(i==0){
		return (const Dtype*)(thrust::raw_pointer_cast(_gpu_vector.data()));
	}else{
		return (const Dtype*)(thrust::raw_pointer_cast(_gpu_vector.data()+i*_ld));
	}
	#else
	__NO_GPU__;
	#endif
}
template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_gpu_data(int i){
	CHECK_GE(i, 0);
	#ifndef CPU_ONLY
	to_gpu();
	_status=GPU_VECTOR;
	if(_gpu_vector.empty()){
		return NULL;
	}
	if(i==0){
		return (Dtype*)(thrust::raw_pointer_cast(_gpu_vector.data()));
	}else{
		return (Dtype*)(thrust::raw_pointer_cast(_gpu_vector.data()+i*_ld));
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
int SyncVector<Dtype>::ld(){
	return _ld;
}

template<typename Dtype>
int SyncVector<Dtype>::size(){
	int s=0;
	switch(_status){
	case GPU_VECTOR:
		s=_gpu_vector.size();
		break;
	case UNINITIALIZED:
	case CPU_VECTOR:
	case SYN_VECTOR:
	default:
		s=_cpu_vector.size();
		break;
	}
	return s;
}

#ifdef CPU_ONLY
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

template <typename Dtype>
VectorStateI8 SyncVector<Dtype>::state_i8(){
	VectorStateI8 vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const int8_t)((*v)[i]));
	}
	return vs;
}
template <typename Dtype>
VectorStateI16 SyncVector<Dtype>::state_i16(){
	VectorStateI16 vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const int16_t)((*v)[i]));
	}
	return vs;
}
template <typename Dtype>
VectorStateI32 SyncVector<Dtype>::state_i32(){
	VectorStateI32 vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const int32_t)((*v)[i]));
	}
	return vs;
}
template <typename Dtype>
VectorStateI64 SyncVector<Dtype>::state_i64(){
	VectorStateI64 vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const int64_t)((*v)[i]));
	}
	return vs;
}
template <typename Dtype>
VectorStateF16 SyncVector<Dtype>::state_f16(){
	VectorStateF16 vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const float)(fp16_to_fp32((*v)[i])));
	}
	return vs;
}
template <typename Dtype>
VectorStateF32 SyncVector<Dtype>::state_f32(){
	VectorStateF32 vs;
	vs.set_ld(_ld);
	CONST_HOST_VECTOR(Dtype, *v)=cpu_vector();
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const float)((*v)[i]));
	}
	return vs;
}
template <typename Dtype>
VectorStateF64 SyncVector<Dtype>::state_f64(){
	VectorStateF64 vs;
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

template class SyncVector<int8_t>;
template class SyncVector<int16_t>;
template class SyncVector<int32_t>;
template class SyncVector<int64_t>;
template class SyncVector<fp16>;
template class SyncVector<float>;
template class SyncVector<double>;

}
