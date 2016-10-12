#include "gsbn/Blob.hpp"

namespace gsbn{

Blob::Blob(const string name, vector<int>& shape) :
	_name(name),
	_data(new MemBlock()){
	
	int dim = shape.size();
	CHECK_GT(dim, 0) << "Blob should have dim > 0!";
	int _count=0;
	for(int i=0; i<dim; i++){
		_count*=shape[i];
	}
	CHECK_GT(_count, 0) << "Blob should have some data!";
	_shape = shape;
	MemBlock* new_data_ptr = new MemBlock(c*sizeof(DType));
	_data = new_data_ptr;
}

void Blob::reshape(vector<int>& shape){
	int dim = shape.size();
	CHECK_GT(dim, 0) << "Blob should have dim > 0!";
	int c=0;
	for(int i=0; i<dim; i++){
		c*=shape[i];
	}
	CHECK_EQ(c, _count) << "Blob shapes don't match!";
	_shape=shape;
}

const string Blob::shape_str(){
	
}

const vector<int> Blob::shape(){
	return (const vector<int>)(_shape);
}

int Blob::offset(vector<int>& indices){
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
		size=1;
		for(int j=i+1; j<blob_dim; j++){
			size *= _shape[j];
		}
		size *= indices[i];
		os += size;
	}
	
	return os;
}

const DType Blob::cpu_data(vector<int>& indices){
	return static_cast<const DType*>(_data->cpu_data())+offset(indices);
}
const DType Blob::gpu_data(vector<int>& indices){
	return static_cast<const DType*>(_data->gpu_data())+offset(indices);
}
DType Blob::mutable_cpu_data(vector<int>& indices){
	return static_cast<DType*>(_data->mutable_cpu_data())+offset(indices);
}
DType Blob::mutable_gpu_data(vector<int>& indices){
	return static_cast<DType*>(_data->mutable_gpu_data())+offset(indices);
}

}
