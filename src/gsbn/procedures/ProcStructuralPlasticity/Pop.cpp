#include "gsbn/procedures/ProcStructuralPlasticity/Pop.hpp"

namespace gsbn{
namespace proc_structural_plasticity{

Pop::Pop(int& id, int& hcu_start, int& mcu_start, PopParam pop_param, Database& db, int rank){
	_id = id;
	_rank = pop_param.rank();
	_dim_hcu = pop_param.hcu_num();
	_dim_mcu = pop_param.mcu_num();
	_hcu_start = hcu_start;
	_mcu_start = mcu_start;
	
	for(int k=0; k<pop_param.shape_size(); k++){
		_shape.push_back(pop_param.shape(k));
	}
	for(int k=0; k<pop_param.position_size(); k++){
		_position.push_back(pop_param.position(k));
	}
	
	// DO NOT CHECK THE RETURN VALUE, SINCE THE SPIKE VECTOR MAYBE NOT IN THE CURRENT
	// RANK.
	_act = db.sync_vector_f32("act_" + to_string(_id));
	
	_hcu_start = hcu_start;
	_mcu_start = mcu_start;
	hcu_start += _dim_hcu;
	mcu_start += _dim_hcu * _dim_mcu;
	
	if(_rank == rank){
		CHECK(_fanout = db.create_sync_vector_i32(".fanout_" + to_string(_id)));
		_fanout->resize(_dim_hcu*_dim_mcu, pop_param.fanout_num());
	}else{
		_fanout=NULL;
	}
	
	id++;
}

Pop::~Pop(){
}

void Pop::add_prj(int prj_id){
	_avail_prj_list.push_back(prj_id);
}

vector<int> Pop::get_avail_prj_list(){
	return _avail_prj_list;
}

vector<int> Pop::get_avail_active_mcu_list(){
	vector<int> v(_dim_hcu*_dim_mcu);
	iota(v.begin(), v.end(), 0);
	return v;
}

vector<int> Pop::hcu_coor(int hcu_idx){
	vector<int> coor(_shape.size());
	int index=hcu_idx;
	for(int i=0; i<_shape.size(); i++){
		int d = _shape[i];
		CHECK_GT(d, 0);
		coor[i] = index % d;
		index /= d;
	}
	if(coor.size()>=_position.size()){
		for(int i=0; i<_position.size(); i++){
			coor[i] += _position[i];
		}
	}else{
		int size = coor.size();
		for(int i=0; i<_position.size(); i++){
			if(i<size){
				coor[i] += _position[i];
			}else{
				coor.push_back(_position[i]);
			}
		}
	}
	return coor;
}

}
}
