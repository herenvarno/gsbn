#include "gsbn/SyncVector.hpp"

namespace gsbn{

#ifdef CPU_ONLY


template <typename Dtype>
SyncVector<Dtype>::SyncVector(): _ld(0), _vectors(){

	vec_t vec0={
		._t=HOST,
		._s=UP_TO_DATE
	};
	
	_vectors[0]=vec0;
}
template <typename Dtype>
SyncVector<Dtype>::~SyncVector(){
}

template <typename Dtype>
void SyncVector<Dtype>::register_device(int id, type_t type){
	CHECK_GE(id, 0) << "Wrong ID.";
	if(_vectors.find(id)==_vectors.end()){
		LOG(WARNING) << "Device vector already exist!";
		return;
	}
	
	vec_t vecx={
		._t=type,
		._s=OUT_OF_DATE
	};
	
	_vectors[id]=vecx;
}

template <typename Dtype>
const std::vector<Dtype>* SyncVector<Dtype>::host_vector(int id){
	to_device(id);
	auto it = _vectors.find(id);
	CHECK_NE(it, _vectors.end());
	CHECK_EQ(it->second._t, HOST)<< "Vector type is wrong!";
	return (const std::vector<Dtype>*)(&(it->second._v_host));
}
template <typename Dtype>
std::vector<Dtype>* SyncVector<Dtype>::mutable_host_vector(int id){
	to_device(id);
	auto it = _vectors.find(id);
	CHECK_NE(it, _vectors.end());
	CHECK_EQ(it->second._t, HOST)<< "Vector type is wrong!";
	
	for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
		if(itx->first==id){
			continue;
		}
		itx->second._s=OUT_OF_DATE;
	}
	
	return (std::vector<Dtype>*)(&(it->second._v_host));
}

template <typename Dtype>
const Dtype* SyncVector<Dtype>::host_data(int id, int i){
	CHECK_GE(i, 0);
	
	const std::vector<Dtype>* vecx = host_vector(id);
	
	if(vecx->empty()){
		return NULL;
	}
	
	if(i==0){
		return (const Dtype*)(&((*vecx)[0]));
	}else if(_ld<=0){
		return (const Dtype*)(&((*vecx)[i]));
	}else{
		return (const Dtype*)(&((*vecx)[i*_ld]));
	}
}

template <typename Dtype>
Dtype* SyncVector<Dtype>::mutable_host_data(int id, int i){
	CHECK_GE(i, 0);
	
	std::vector<Dtype>* vecx = mutable_host_vector(id);
	
	if(vecx->empty()){
		return NULL;
	}
	
	if(i==0){
		return (Dtype*)(&((*vecx)[0]));
	}else if(_ld<=0){
		return (Dtype*)(&((*vecx)[i]));
	}else{
		return (Dtype*)(&((*vecx)[i*_ld]));
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
			s=itx->second._v_host.size();
			break;
		}
	}
	return s;
}

template<typename Dtype>
void SyncVector<Dtype>::resize(size_t s, Dtype val, bool lazy){
	bool updated=false;
	for(auto itx=_vectors.begin(); itx!=_vectors.end(); itx++){
		if(itx->second._s==UP_TO_DATE){
			if(updated==false || lazy==false){
				itx->second._v_host.resize(s, val);
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
				itx->second._v_host.push_back(val);
			}else{
				itx->second._s==OUT_OF_DATE;
			}
		}
	}
}

template <typename Dtype>
VectorStateI8 SyncVector<Dtype>::state_i8(){
	VectorStateI8 vs;
	vs.set_ld(_ld);
	const std::vector<Dtype> *v = host_vector(0);
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
	const std::vector<Dtype> *v = host_vector(0);
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
	const std::vector<Dtype> *v = host_vector(0);
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
	const std::vector<Dtype> *v = host_vector(0);
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
	const std::vector<Dtype> *v = host_vector(0);
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
	const std::vector<Dtype> *v = host_vector(0);
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
	const std::vector<Dtype> *v = host_vector(0);
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
				it->second._v_host = itx->second._v_host;
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
				it0->second._v_host = itx->second._v_host;
				(it0->second)._s = UP_TO_DATE;
				break;
			}
		}
	}
	it->second._v_host = it0->second._v_host;
	(it->second)._s = UP_TO_DATE;
}


/*
 * Deprecated
 */
template <typename Dtype>
const std::vector<Dtype>* SyncVector<Dtype>::cpu_vector(){
	return host_vector(0);
}
template <typename Dtype>
std::vector<Dtype>* SyncVector<Dtype>::mutable_cpu_vector(){
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




#else



#endif


template class SyncVector<int8_t>;
template class SyncVector<int16_t>;
template class SyncVector<int32_t>;
template class SyncVector<int64_t>;
template class SyncVector<fp16>;
template class SyncVector<float>;
template class SyncVector<double>;

}
