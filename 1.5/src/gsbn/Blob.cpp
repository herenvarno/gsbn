#include "gsbn/Blob.hpp"

namespace gsbn{

template <typename Dtype>
Blob<Dtype>::Blob(const string name, const vector<int>& shape) :
	_name(name){
	
	int dim = shape.size();
	CHECK_GT(dim, 0) << "Blob should have dim > 0!";
	_count=0;
	for(int i=0; i<dim; i++){
		_count*=shape[i];
	}
	CHECK_GT(_count, 0) << "Blob should have some data!";
	_shape = shape;
	MemBlock* new_data_ptr = new MemBlock(_count*sizeof(Dtype));
	_data = new_data_ptr;
}

template <typename Dtype>
void Blob<Dtype>::reshape(const vector<int>& shape){
	int dim = shape.size();
	CHECK_GT(dim, 0) << "Blob should have dim > 0!";
	int c=0;
	for(int i=0; i<dim; i++){
		c*=shape[i];
	}
	CHECK_EQ(c, _count) << "Blob shapes don't match!";
	_shape=shape;
}

template <typename Dtype>
const string Blob<Dtype>::shape_str(){
	
}

template <typename Dtype>
const vector<int> Blob<Dtype>::shape(){
	return (const vector<int>)(_shape);
}

template <typename Dtype>
int Blob<Dtype>::offset(const vector<int>& indices){
	int d=indices.size();
	CHECK_GE(d, 0);
	if(d==0)
		return 0;
	
	int blob_dim = _shape.size();
	vector<int> size_dim = _shape;
	int os=0;
	for(int i=0; i<blob_dim; i++){
		if(i>d){
			break;
		}
		int size=1;
		for(int j=i+1; j<blob_dim; j++){
			size *= _shape[j];
		}
		size *= indices[i];
		os += size;
	}
	
	return os;
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data(const vector<int>& indices){
	return static_cast<const Dtype*>(_data->cpu_data())+offset(indices);
}
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data(const vector<int>& indices){
	return static_cast<const Dtype*>(_data->gpu_data())+offset(indices);
}
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data(const vector<int>& indices){
	return static_cast<Dtype*>(_data->mutable_cpu_data())+offset(indices);
}
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data(const vector<int>& indices){
	return static_cast<Dtype*>(_data->mutable_gpu_data())+offset(indices);
}

template <typename Dtype>
int Blob<Dtype>::dim(){
	return _shape.size();
}

template <typename Dtype>
int Blob<Dtype>::count(){
	return _count;
}

template <typename Dtype>
const string Blob<Dtype>::name(){
	return (const string)(_name);
}

template <typename Dtype>
const shared_ptr<MemBlock>& Blob<Dtype>::data(){
	return (const shared_ptr<MemBlock> &) _data;
}

}
