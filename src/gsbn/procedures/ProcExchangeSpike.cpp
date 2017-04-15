#include "gsbn/procedures/ProcExchangeSpike.hpp"

namespace gsbn{
namespace proc_exchange_spike{

REGISTERIMPL(ProcExchangeSpike);

void ProcExchangeSpike::init_new(SolverParam solver_param, Database& db){
	_db = &db;
	
	CHECK(_glv.geti("rank", _rank));
	CHECK(_glv.geti("num-rank", _num_rank));
	
	NetParam net_param = solver_param.net_param();
	ProcParam proc_param = get_proc_param(solver_param);
	
	int pop_id=0;
	int hcu_cnt=0;
	int mcu_cnt=0;
	_spike_buffer_size = 1;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			int pop_rank = pop_param.rank();
			CHECK_LT(pop_rank, _num_rank);
			_pop_rank.push_back(pop_rank);
			int dim_hcu = pop_param.hcu_num();
			int dim_mcu = pop_param.mcu_num();
			_pop_num_element.push_back(dim_hcu*dim_mcu);
			_pop_shared_offset.push_back(mcu_cnt);
			
			if(pop_rank == _rank){
				mcu_cnt += (dim_hcu * dim_mcu);
			}
			
			SyncVector<int8_t> *spike=NULL;
			if(pop_rank == _rank){
				spike = db.sync_vector_i8("spike_"+to_string(pop_id));
				CHECK(spike);
				if(spike->ld()>0){
					if(_spike_buffer_size!=1){
						CHECK_EQ(_spike_buffer_size, spike->size()/spike->ld());
					}else{
						_spike_buffer_size = spike->size()/spike->ld();
					}
				}
				CHECK_EQ(spike->size(), _spike_buffer_size * dim_hcu*dim_mcu);
			}
			_pop_sj.push_back(spike);
			
			pop_id++;
		}
	}
	
	_shared_buffer.resize(mcu_cnt);
	
	int proj_id=0;
	int total_pop_num = _pop_sj.size();
	int proj_param_size = net_param.proj_param_size();
	
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			SyncVector<int8_t> *si=NULL;
			
			if(_pop_rank[dest_pop]!=_rank){
				dest_pop=-1;
			}else{
				int num_element = _pop_num_element[dest_pop];
				si = db.sync_vector_i8("si_"+to_string(proj_id));
				CHECK(si);
				CHECK_EQ(si->size(), num_element);
			}
			
			_proj_src_pop.push_back(src_pop);
			_proj_dest_pop.push_back(dest_pop);
			_proj_si.push_back(si);
			
			proj_id++;
		}
	}
	
	MPI_Win_create(&_shared_buffer[0], _shared_buffer.size(), sizeof(int8_t), MPI_INFO_NULL, MPI_COMM_WORLD, &_win);
	
}
void ProcExchangeSpike::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcExchangeSpike::update_cpu(){
	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if(cycle_flag < 0){
		MPI_Win_fence(0, _win);
		MPI_Win_free(&_win);
		return;
	}else if(cycle_flag!=1){
		return;
	}
	
	int simstep;
	CHECK(_glv.geti("simstep", simstep));
	
	// COPY SJ TO SHARED MEMORY
	for(int i=0; i<_proj_dest_pop.size(); i++){
		int pop = _proj_dest_pop[i];
		if(pop<0){
			continue;
		}
		int rank = _pop_rank[pop];
		int shared_offset = _pop_shared_offset[pop];
		int num_element = _pop_num_element[pop];
		const int8_t *ptr_sj = _pop_sj[pop]->cpu_data()+(simstep%_spike_buffer_size)*num_element;
		memcpy(&_shared_buffer[shared_offset], ptr_sj, num_element*sizeof(int8_t));
	}
	
	MPI_Win_fence(0, _win);
	// DISTRIBUTE SPIKES TO SI
	for(int i=0; i<_proj_src_pop.size(); i++){
		if(_proj_dest_pop[i]<0){
			continue;
		}
		int pop = _proj_src_pop[i];
		int rank = _pop_rank[pop];
		int shared_offset = _pop_shared_offset[pop];
		int num_element = _pop_num_element[pop];
		int8_t *ptr_si = _proj_si[i]->mutable_cpu_data();
		MPI_Get(ptr_si, num_element, MPI_CHAR, rank, shared_offset, num_element, MPI_CHAR, _win);
	}
	MPI_Win_fence(0, _win);
}

#ifndef CPU_ONLY
void ProcExchangeSpike::update_gpu(){
	update_cpu();
}
#endif

}
}
