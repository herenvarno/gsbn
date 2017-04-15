#include "gsbn/procedures/ProcCheck/Pop.hpp"

namespace gsbn{
namespace proc_check{

Pop::Pop(int& id, int& hcu_start, int& mcu_start, PopParam pop_param, Database& db){
	_id = id;
	_rank = pop_param.rank();
	_dim_hcu = pop_param.hcu_num();
	_dim_mcu = pop_param.mcu_num();
	
	_counter.resize(_dim_hcu);
	#pragma omp parallel for
	for(int i=0; i<_dim_hcu; i++){
		_counter[i].resize(_dim_mcu);
	}
	_active_mcu_idx.resize(_dim_hcu);
	_active_mcu_cnt.resize(_dim_hcu);
	
	// DO NOT CHECK THE RETURN VALUE, SINCE THE SPIKE VECTOR MAYBE NOT IN THE CURRENT
	// RANK.
	_spike = db.sync_vector_i8("spike_" + to_string(_id));
	
	_hcu_start = hcu_start;
	_mcu_start = mcu_start;
	hcu_start += _dim_hcu;
	mcu_start += _dim_hcu * _dim_mcu;
	
	id++;
}

Pop::~Pop(){
}

void Pop::update_counter(int cursor){
	const int8_t *ptr_spike = _spike->cpu_data()+cursor*_dim_hcu*_dim_mcu;
	#pragma omp parallel for
	for(int i=0; i<_dim_hcu; i++){
		for(int j=0; j<_dim_mcu; j++){
			_counter[i][j] += int(ptr_spike[i*_dim_mcu+j]>0);
		}
	}
}

void Pop::check_result(){
	#pragma omp parallel for
	for(int i=0; i<_dim_hcu; i++){
		int max_idx = -1;
		int max_cnt = 0;
		for(int j=0; j<_dim_mcu; j++){
			if(_counter[i][j]>max_cnt){
				max_idx = j;
				max_cnt = _counter[i][j];
			}
			_counter[i][j] = 0;
		}
		_active_mcu_idx[i] = max_idx;
		_active_mcu_cnt[i] = max_cnt;
	}
}

void Pop::send_result(){
	MPI_Put(&_active_mcu_idx[0], _dim_hcu, MPI_INT, 0, _hcu_start, _dim_hcu, MPI_INT, _win_idx);
	MPI_Put(&_active_mcu_cnt[0], _dim_hcu, MPI_INT, 0, _hcu_start, _dim_hcu, MPI_INT, _win_cnt);
}

void Pop::set_win_idx(MPI_Win win_idx){
	_win_idx = win_idx;
}

void Pop::set_win_cnt(MPI_Win win_cnt){
	_win_cnt = win_cnt;
}

}
}
