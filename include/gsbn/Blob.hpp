#ifndef __GSBN_BLOB_HPP__
#define __GSBN_BLOB_HPP__

#include "gsbn/Common.hpp"
#include "gsbn/MemBlock.hpp"

namespace gsbn{

/**
 * \class Blob
 * \bref Manage memory block data in both main memory and GPU global memory.
 *
 * \warning Class Blob is deprecated and will be removed in the future. Use SyncVector
 * to manage your data storage.
 */

template <typename Dtype>
class Blob{

public:
	Blob(const string name, const vector<int>& shape);
	
	void reshape(const vector<int>& shape);
	inline const string shape_str();
	inline const vector<int> shape();
	
	inline int offset(const vector<int>& indices={});
	
	const Dtype* cpu_data(const vector<int>& indices={});
	const Dtype* gpu_data(const vector<int>& indices={});
	Dtype* mutable_cpu_data(const vector<int>& indices={});
	Dtype* mutable_gpu_data(const vector<int>& indices={});
	
	inline int dim();
	inline int count();
	inline const string name();
	inline const shared_ptr<MemBlock>& data();
	
protected:
	string _name;
	vector<int> _shape;
	int _count;
	shared_ptr<MemBlock> _data;
};

}

#endif // __GSBN_BLOB_HPP__
