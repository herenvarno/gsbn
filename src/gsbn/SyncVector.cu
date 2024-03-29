#include "gsbn/SyncVector.hpp"

namespace gsbn{

#ifndef CPU_ONLY


template <typename Dtype>
SyncVector<Dtype>::SyncVector(): _ld(0), _vectors(){

	vec_t vec0={
		._t=HOST,
		._s=UP_TO_DATE
	};
	vec_t vec1={
		._t=DEVICE,
		._s=UP_TO_DATE
	};
	
	_vectors[0]=vec0;
	_vectors[1]=vec1;
}
template <typename Dtype>
SyncVector<Dtype>::~SyncVector(){
}

template <typename Dtype>
void SyncVector<Dtype>::register_device(int id, type_t type){
	CHECK_GE(id, 0) << "Wrong ID.";
	if(_vectors.find(id)==_vectors.end()){
		LOG(WARNING) << "Device vector already exist!";
	}
	
	vec_t vecx={
		._t=type,
		._s=OUT_OF_DATE
	};
	
	_vectors[id]=vecx;
}

template <typename Dtype>
const thrust::host_vector<Dtype>* SyncVector<Dtype>::host_vector(int id){
	to_device(id);
	auto it = _vectors.find(id);
	CHECK_NE(it, _vectors.end());
	CHECK_EQ(it->second._t, HOST)<< "Vector type is wrong!";
	return (const thrust::host_vector<Dtype>*)(&(it->second._v_host));
}
template <typename Dtype>
thrust::host_vector<Dtype>* SyncVector<Dtype>::mutable_host_vector(int id){
	to_device(id);
	auto it = _vectors.find(id);
	CHECK_NE(it, _vectors.end());
	CHECK_EQ(it->second._t, HOST) << "Vector type is wrong!";
	
	for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
		if(itx->first==id){
			continue;
		}
		itx->second._s=OUT_OF_DATE;
	}
	
	return (thrust::host_vector<Dtype>*)(&(it->second._v_host));
}

template <typename Dtype>
const Dtype* SyncVector<Dtype>::host_data(int id, int i){
	CHECK_GE(i, 0);
	
	const thrust::host_vector<Dtype>* vecx = host_vector(id);
	
	if(vecx->empty()){
		return NULL;
	}
	
	if(i==0){
		return (const Dtype*)(thrust::raw_pointer_cast(vecx->data()));
	}else if(_ld<=0){
		return (const Dtype*)(thrust::raw_pointer_cast(vecx->data())+i);
	}else{
		return (const Dtype*)(thrust::raw_pointer_cast(vecx->data())+i*_ld);
	}
}

template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_host_data(int id, int i){
	CHECK_GE(i, 0);
	
	thrust::host_vector<Dtype>* vecx = mutable_host_vector(id);
	
	if(vecx->empty()){
		return NULL;
	}
	
	if(i==0){
		return (Dtype*)(thrust::raw_pointer_cast(vecx->data()));
	}else if(_ld<=0){
		return (Dtype*)(thrust::raw_pointer_cast(vecx->data())+i);
	}else{
		return (Dtype*)(thrust::raw_pointer_cast(vecx->data())+i*_ld);
	}
}

template <typename Dtype>
const thrust::device_vector<Dtype>* SyncVector<Dtype>::device_vector(int id){
	to_device(id);
	auto it = _vectors.find(id);
	CHECK_NE(it, _vectors.end());
	CHECK_EQ(it->second._t, DEVICE)<< "Vector type is wrong!";
	return (const thrust::device_vector<Dtype>*)(&(it->second._v_device));
}
template <typename Dtype>
thrust::device_vector<Dtype>* SyncVector<Dtype>::mutable_device_vector(int id){
	to_device(id);
	auto it = _vectors.find(id);
	CHECK_NE(it, _vectors.end());
	CHECK_EQ(it->second._t, DEVICE)<< "Vector type is wrong!";
	
	for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
		if(itx->first==id){
			continue;
		}
		itx->second._s=OUT_OF_DATE;
	}
	
	return (thrust::device_vector<Dtype>*)(&(it->second._v_device));
}

template <typename Dtype>
const Dtype* SyncVector<Dtype>::device_data(int id, int i){
	CHECK_GE(i, 0);
	
	const thrust::device_vector<Dtype>* vecx = device_vector(id);
	
	if(vecx->empty()){
		return NULL;
	}
	
	if(i==0){
		return (const Dtype*)(thrust::raw_pointer_cast(vecx->data()));
	}else if(_ld<=0){
		return (const Dtype*)(thrust::raw_pointer_cast(vecx->data())+i);
	}else{
		return (const Dtype*)(thrust::raw_pointer_cast(vecx->data())+i*_ld);
	}
}

template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_device_data(int id, int i){
	CHECK_GE(i, 0);
	
	thrust::device_vector<Dtype>* vecx = mutable_device_vector(id);
	
	if(vecx->empty()){
		return NULL;
	}
	
	if(i==0){
		return (Dtype*)(thrust::raw_pointer_cast(vecx->data()));
	}else if(_ld<=0){
		return (Dtype*)(thrust::raw_pointer_cast(vecx->data())+i);
	}else{
		return (Dtype*)(thrust::raw_pointer_cast(vecx->data())+i*_ld);
	}
}

template <typename Dtype>
void SyncVector<Dtype>::set_ld(int l){
	CHECK_GE(l, 0);
	_ld = l;
}

template <typename Dtype>
int SyncVector<Dtype>::ld(){
	return _ld;
}

template<typename Dtype>
int SyncVector<Dtype>::size(){
	int s=0;
	for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
		if(itx->second._s==UP_TO_DATE){
			switch(itx->second._t){
			case HOST:
				s=itx->second._v_host.size();
				break;
			case DEVICE:
				s=itx->second._v_device.size();
				break;
			}
			break;
		}
	}
	return s;
}

template <typename Dtype>
VectorStateI8 SyncVector<Dtype>::state_i8(){
	VectorStateI8 vs;
	vs.set_ld(_ld);
	const thrust::host_vector<Dtype> *v = host_vector(0);
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
	const thrust::host_vector<Dtype> *v = host_vector(0);
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
	const thrust::host_vector<Dtype> *v = host_vector(0);
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
	const thrust::host_vector<Dtype> *v = host_vector(0);
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
	const thrust::host_vector<Dtype> *v = host_vector(0);
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
	const thrust::host_vector<Dtype> *v = host_vector(0);
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
	const thrust::host_vector<Dtype> *v = host_vector(0);
	int size=v->size();
	for(int i=0; i<size; i++){
		vs.add_data((const double)((*v)[i]));
	}
	return vs;
}

template <typename Dtype>
void SyncVector<Dtype>::to_device(int id){
	CHECK_GE(id, 0);
	
	auto it=_vectors.find(id);
	if(it==_vectors.end()){
		LOG(FATAL) << "Vector on device "<< id <<" is not registered!";
	}
	
	if((it->second)._s==UP_TO_DATE){
		return;
	}
	
	if(id==0){
		for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
			if(itx->first==0){
				continue;
			}
			if((itx->second)._s==UP_TO_DATE){
				assign_vec(it->second, itx->second);
				(it->second)._s = UP_TO_DATE;
				break;
			}
		}
		return;
	}
	
	auto it0 = _vectors.find(0);
	if(it==_vectors.end()){
		LOG(FATAL) << "Vector on device \"0\" (HOST) is not registered!";
	}
	
	if((it0->second)._s!=UP_TO_DATE){
		for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
			if(itx->first==0){
				continue;
			}
			if((itx->second)._s==UP_TO_DATE){
				assign_vec(it0->second, itx->second);
				(it0->second)._s = UP_TO_DATE;
				break;
			}
		}
	}
	assign_vec(it->second, it0->second);
	(it->second)._s = UP_TO_DATE;
}

template <typename Dtype>
void SyncVector<Dtype>::assign_vec(vec_t& a, vec_t& b){
	switch(a._t){
	case HOST:
		switch(b._t){
		case HOST:
			a._v_host = b._v_host;
			break;
		case DEVICE:
			a._v_host = b._v_device;
			break;
		default:
			break;
		}
		break;
	case DEVICE:
		switch(b._t){
		case HOST:
			a._v_device = b._v_host;
			break;
		case DEVICE:
			a._v_device = b._v_device;
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}
}



/*
 * Deprecated
 */
template <typename Dtype>
const thrust::host_vector<Dtype>* SyncVector<Dtype>::cpu_vector(){
	return host_vector(0);
}
template <typename Dtype>
thrust::host_vector<Dtype>* SyncVector<Dtype>::mutable_cpu_vector(){
	return mutable_host_vector(0);
}
template <typename Dtype>
const Dtype* SyncVector<Dtype>::cpu_data(int i){
	return host_data(0, i);
}
template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_cpu_data(int i){
	return mutable_host_data(0, i);
}
template <typename Dtype>
const thrust::device_vector<Dtype>* SyncVector<Dtype>::gpu_vector(){
	return device_vector(1);
}
template <typename Dtype>
thrust::device_vector<Dtype>* SyncVector<Dtype>::mutable_gpu_vector(){
	return mutable_device_vector(1);
}
template <typename Dtype>
const Dtype* SyncVector<Dtype>::gpu_data(int i){
	return device_data(1,i);
}
template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_gpu_data(int i){
	return mutable_device_data(1,i);
}

template <typename Dtype>
void SyncVector<Dtype>::print_state(){
	int i=0;
	for(auto it=_vectors.begin(); it!=_vectors.end(); it++){
		cout << i << ":" << it->second._s << "," << it->second._t << endl;
		i++;	
	}
}

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



template class SyncVector<int8_t>;
template class SyncVector<int16_t>;
template class SyncVector<int32_t>;
template class SyncVector<int64_t>;
template class SyncVector<fp16>;
template class SyncVector<float>;
template class SyncVector<double>;
#endif



}
