#include "gsbn/procedures/ProcStructuralPlasticity.hpp"

namespace gsbn{
namespace proc_structural_plasticity{

REGISTERIMPL(ProcStructuralPlasticity);

void ProcStructuralPlasticity::init_new(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	CHECK(_glv.getf("dt", _dt));
	int rank;
	CHECK(_glv.geti("rank", rank));
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	Parser par(proc_param);
	_d_norm = 0.75;
	if(par.argf("d-norm", _d_norm)){
		if(_d_norm<0){
			_d_norm = 0;
		}
	}
	_v_cond = 0.2;
	if(par.argf("v-cond", _v_cond)){
		if(_v_cond<=0){
			_v_cond = 1;
		}
	}
	_t_th = 100;
	if(par.argi("t-th", _t_th)){
		if(_t_th<=0){
			_t_th = 100;
		}
	}
	_period = 100;
	if(par.argi("period", _period)){
		if(_period<=0){
			_period = 100;
		}
	}
	_pruning_period = 1000;
	if(par.argi("pruning-period", _pruning_period)){
		if(_pruning_period<=0){
			_pruning_period = 1000;
		}
	}
	_enable_geometry = 0;
	if(par.argi("enable-geometry", _enable_geometry)){
		if(_enable_geometry!=0){
			_enable_geometry = 1;
		}
	}
	
	_wp = 0.5;
	par.argf("wp", _wp);
	_wn = -0.5;
	par.argf("wn", _wn);
	
	int num_rank;
	CHECK(_glv.geti("num-rank", num_rank));
	
	vector<int> shared_buffer_size_list(num_rank);
	
	int pop_id=0;
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop p(pop_id, hcu_cnt, mcu_cnt, pop_param, db, rank);
			_pop_list.push_back(p);
		}
	}
	
	int proj_id=0;
	int total_pop_num = pop_id;
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			Prj prj(proj_id, shared_buffer_size_list, _pop_list, proj_param, db, rank);
			_pop_list[src_pop].add_prj(prj._id);
			_prj_list.push_back(prj);
		}
	}
	
	_shared_buffer.resize(shared_buffer_size_list[rank]);
	MPI_Win_create(&_shared_buffer[0], _shared_buffer.size(), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &_win);
	
	// DEBUG INITIALIZE ALL THE CONNECTIONS
//	for(int i=0; i<_prj_list.size(); i++){
//		Prj prj=_prj_list[i];
//		if(prj._rank != rank){
//			continue;
//		}
//		const int *ptr_ii = prj._ii->cpu_data();
//		for(int j=0; j<prj._dim_hcu; j++){
//			for(int k=0; k<prj._dim_conn; k++){
//				int index = j*prj._dim_conn+k;
//				Pop src_pop = _pop_list[prj._src_pop];
//				Pop dest_pop = _pop_list[prj._dest_pop];
//				int src_mcu = k;
//				int src_hcu = k/src_pop._dim_mcu;
//				int delay = 1;
//				if(_enable_geometry){
//					delay = delay_cycle(i, src_mcu, j);
//				}
//				add_row(i, src_mcu, j, delay);
//			}
//		}
//	}
	
	
}

void ProcStructuralPlasticity::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcStructuralPlasticity::update_cpu(){

	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if(cycle_flag < 0){
		MPI_Win_fence(0, _win);
		MPI_Win_free(&_win);
		return;
	}else if(cycle_flag!=1){
		return;
	}
	
	bool plasticity;
	CHECK(_glv.getb("plasticity", plasticity));
	if(!plasticity){
		return;
	}
	
	int rank;
	CHECK(_glv.geti("rank", rank));
	int simstep;
	CHECK(_glv.geti("simstep", simstep));
	float dt;
	CHECK(_glv.getf("dt", dt));
	int maxfq = 100;
	
	if(simstep%_pruning_period==0){
//	LOG(INFO) << "PROC STRUCT PLASTICITY UPDATE STAGE 1!";
	// STAGE 1: REMOVE CONNECTION
	for(int i=0; i<_prj_list.size(); i++){
		Prj prj=_prj_list[i];
		if(prj._rank != rank){
			continue;
		}
		vector<int>v = prj.prune(simstep-_t_th, _wp, _wn);
		//std::copy(v.begin(), v.end(), _shared_buffer.begin()+prj._shared_buffer_offset);
		for(int j=0; j<prj._dim_hcu * prj._dim_conn; j++){
			_shared_buffer[prj._shared_buffer_offset + j] = v[j];
			if(v[j]<0){
				break;
			}
		}
//		const int *ptr_ii = prj._ii->cpu_data();
//		const int *ptr_ti = prj._ti->cpu_data();
//		int idx=0;
//		_shared_buffer[prj._shared_buffer_offset + idx] = -1;
//		for(int j=0; j<prj._dim_hcu * prj._dim_conn; j++){
//			if(((ptr_ti[j]<0) || (simstep - _t_th > ptr_ti[j])) && (ptr_ii[j]>=0)){
//				prj.remove_conn(j);
//				_shared_buffer[prj._shared_buffer_offset + idx] = ptr_ii[j];
//				idx ++;
//				if(idx < prj._dim_hcu * prj._dim_conn){
//					_shared_buffer[prj._shared_buffer_offset + idx] = -1;
//				}
//			}
//		}
	}
	
//		if(simstep==2000)
//	for(int i=0; i<_prj_list.size(); i++){
//	Prj prj=_prj_list[i];
//	for(int j=0; j<prj._dim_hcu; j++){
//		for(int k=0; k<prj._dim_conn; k++){
//			int index = j*prj._dim_conn+k;
//			cout << _shared_buffer[prj._shared_buffer_offset+index] << ",";
//		}
//	}
//	cout << endl;
//	}
	//LOG(INFO) << "PROC STRUCT PLASTICITY UPDATE STAGE 2!";
	// STAGE 2: UPDATE FANOUT
	MPI_Win_fence(0, _win);
	for(int i=0; i<_pop_list.size(); i++){
		Pop pop=_pop_list[i];
		if(pop._rank != rank){
			continue;
		}
		for(int j=0; j<pop._avail_prj_list.size(); j++){
			Prj prj = _prj_list[pop._avail_prj_list[j]];
			int *ptr_local_buffer = prj._local_buffer->mutable_cpu_data();
			MPI_Get(ptr_local_buffer, prj._dim_hcu*prj._dim_conn, MPI_INT, prj._rank, prj._shared_buffer_offset, prj._dim_hcu*prj._dim_conn, MPI_INT, _win);
		}
	}
	MPI_Win_fence(0, _win);
	
	for(int i=0; i<_pop_list.size(); i++){
		Pop pop=_pop_list[i];
		if(pop._rank != rank){
			continue;
		}
		int *ptr_fanout = pop._fanout->mutable_cpu_data();
		for(int j=0; j<pop._avail_prj_list.size(); j++){
			Prj prj = _prj_list[pop._avail_prj_list[j]];
			const int *ptr_local_buffer=prj._local_buffer->cpu_data();
			for(int k=0; k<prj._dim_hcu * prj._dim_conn; k++){
				int c = ptr_local_buffer[k];
				if(c<0){
					break;
				}
				ptr_fanout[c] += 1;
			}
		}
	}
	}
	
	if(simstep%_period==0 || simstep==1){
	//LOG(INFO) << "PROC STRUCT PLASTICITY UPDATE STAGE 3!";
	// STAGE 3: PREPARE CURRENT CONNECTION
	for(int i=0; i<_prj_list.size(); i++){
		Prj prj=_prj_list[i];
		if(prj._rank != rank){
			continue;
		}
		const int *ptr_ii = prj._ii->cpu_data();
		memcpy(&(_shared_buffer[prj._shared_buffer_offset]), ptr_ii, prj._dim_hcu*prj._dim_conn*sizeof(int));
	}

	//LOG(INFO) << "PROC STRUCT PLASTICITY UPDATE STAGE 4!";
	// STAGE 4: GET CONNECTION TO SRC SIDE AND BUILD NEW CONNECTION
	
//	for(int i=0; i<_prj_list.size(); i++){
//	Prj prj=_prj_list[i];
//	for(int j=0; j<prj._dim_hcu; j++){
//		for(int k=0; k<prj._dim_conn; k++){
//			int index = j*prj._dim_conn+k;
//			cout << _shared_buffer[prj._shared_buffer_offset+index] << ",";
//		}
//	}
//	cout << endl;
//	}
	
	
	MPI_Win_fence(0, _win);
	for(int i=0; i<_pop_list.size(); i++){
		Pop pop=_pop_list[i];
		if(pop._rank != rank){
			continue;
		}
		for(int j=0; j<pop._avail_prj_list.size(); j++){
			Prj prj = _prj_list[pop._avail_prj_list[j]];
			int *ptr_local_buffer = prj._local_buffer->mutable_cpu_data();
			MPI_Get(ptr_local_buffer, prj._dim_hcu*prj._dim_conn, MPI_INT, prj._rank, prj._shared_buffer_offset, prj._dim_hcu*prj._dim_conn, MPI_INT, _win);
		}
	}
	MPI_Win_fence(0, _win);
	for(int i=0; i<_pop_list.size(); i++){
		Pop pop=_pop_list[i];
		if(pop._rank != rank){
			continue;
		}
		int *ptr_fanout = pop._fanout->mutable_cpu_data();
		
		vector<int> avail_prj_list = pop.get_avail_prj_list();
		vector<int> avail_active_mcu_list = pop.get_avail_active_mcu_list(int(0.2*_period*maxfq*dt));
		while(!avail_prj_list.empty() && !avail_active_mcu_list.empty()){
			float rnd_flt;
			_rnd.gen_uniform01_cpu(&rnd_flt);
			int idx = int(rnd_flt * avail_prj_list.size());
			Prj prj = _prj_list[avail_prj_list[idx]];
			int *ptr_local_buffer = prj._local_buffer->mutable_cpu_data();
			vector<int> short_avail_hcu_list = prj.get_avail_active_hcu_list(0.5);
			if(short_avail_hcu_list.empty()){
				break;
			}
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(short_avail_hcu_list.begin(), short_avail_hcu_list.end(), g);
			for(int j=0; j<short_avail_hcu_list.size(); j++){
				vector<int> short_avail_mcu_list = avail_active_mcu_list;
				for(int k=0; k<prj._dim_conn; k++){
					int c = ptr_local_buffer[short_avail_hcu_list[j]*prj._dim_conn+k];
					if(c>=0){
						short_avail_mcu_list.erase(std::remove(short_avail_mcu_list.begin(), short_avail_mcu_list.end(), c), short_avail_mcu_list.end());
					}
				}
				
				if(short_avail_mcu_list.empty()){
					continue;
				}
				std::random_device rd;
				std::mt19937 g(rd());
				std::shuffle(short_avail_mcu_list.begin(), short_avail_mcu_list.end(), g);
				int l=0;
				for(int k=0; k<prj._dim_conn && l<short_avail_mcu_list.size(); k++){
					int c = ptr_local_buffer[short_avail_hcu_list[j]*prj._dim_conn+k];
					if(c<0){
						ptr_local_buffer[short_avail_hcu_list[j]*prj._dim_conn+k] = short_avail_mcu_list[l];
						l++;
					}
				}
				for(int k=0; k<l; k++){
					int mcu = short_avail_mcu_list[k];
					ptr_fanout[mcu]--;
					if(ptr_fanout[mcu]<=0){
						avail_active_mcu_list.erase(std::remove(avail_active_mcu_list.begin(), avail_active_mcu_list.end(), mcu), avail_active_mcu_list.end());
					}
				}
			}
			avail_prj_list.erase(avail_prj_list.begin()+idx);
		}
	}
	MPI_Win_fence(0, _win);
	for(int i=0; i<_pop_list.size(); i++){
		Pop pop=_pop_list[i];
		if(pop._rank != rank){
			continue;
		}
		for(int j=0; j<pop._avail_prj_list.size(); j++){
			Prj prj = _prj_list[pop._avail_prj_list[j]];
			int *ptr_local_buffer = prj._local_buffer->mutable_cpu_data();
			MPI_Put(ptr_local_buffer, prj._dim_hcu*prj._dim_conn, MPI_INT, prj._rank, prj._shared_buffer_offset, prj._dim_hcu*prj._dim_conn, MPI_INT, _win);
		}
	}
	MPI_Win_fence(0, _win);
	
//	for(int i=0; i<_prj_list.size(); i++){
//	Prj prj=_prj_list[i];
//	for(int j=0; j<prj._dim_hcu; j++){
//		for(int k=0; k<prj._dim_conn; k++){
//			int index = j*prj._dim_conn+k;
//			cout << _shared_buffer[prj._shared_buffer_offset+index] << ",";
//		}
//	}
//	cout << endl;
//	}
//exit(0);
	
	//LOG(INFO) << "PROC STRUCT PLASTICITY UPDATE STAGE 5!";
	// STAGE 5: CREATE NEW CONNECTIONS IN PROJECTION SIDE
	for(int i=0; i<_prj_list.size(); i++){
		Prj prj=_prj_list[i];
		if(prj._rank != rank){
			continue;
		}
		const int *ptr_ii = prj._ii->cpu_data();
		for(int j=0; j<prj._dim_hcu; j++){
			for(int k=0; k<prj._dim_conn; k++){
				int index = j*prj._dim_conn+k;
				
				if(ptr_ii[index]>=0){
					continue;
				}
				if(_shared_buffer[prj._shared_buffer_offset+index]<0){
					continue;
				}
				Pop src_pop = _pop_list[prj._src_pop];
				Pop dest_pop = _pop_list[prj._dest_pop];
				int src_mcu = _shared_buffer[prj._shared_buffer_offset+index];
				int src_hcu = src_mcu/src_pop._dim_mcu;
				int delay = 1;
				if(_enable_geometry){
					delay = delay_cycle(i, src_mcu, j);
				}
				add_row(i, src_mcu, j, delay);
			}
		}
	}
	}
//	exit(0);
}

#ifndef CPU_ONLY
void ProcStructuralPlasticity::update_gpu(){
	update_cpu();
}
#endif

void ProcStructuralPlasticity::add_row(int prj_id, int src_mcu, int dest_hcu, int delay){
	Prj prj = _prj_list[prj_id];
	for(int i=0; i<prj._dim_conn; i++){
		int *ptr_ii = prj._ii->mutable_cpu_data()+dest_hcu*prj._dim_conn;
		int *ptr_di = prj._di->mutable_cpu_data()+dest_hcu*prj._dim_conn;
		if(ptr_ii[i]<0){
			ptr_ii[i]=src_mcu;
			ptr_di[i]=delay;
			break;
		}
	}
}

int ProcStructuralPlasticity::delay_cycle(int prj_idx, int src_mcu, int dest_hcu){
	Prj prj = _prj_list[prj_idx];
	Pop pop0 = _pop_list[prj._src_pop];
	Pop pop1 = _pop_list[prj._dest_pop];
	vector<int> pos0 = pop0.hcu_coor(src_mcu/pop0._dim_mcu);
	vector<int> pos1 = pop1.hcu_coor(dest_hcu);
	
	int size0 = pos0.size();
	int size1 = pos1.size();
	if(size0>size1){
		for(int i=size1; i<size0; i++){
			pos1.push_back(0);
		}
	}else{
		for(int i=size0; i<size1; i++){
			pos0.push_back(0);
		}
		size0 = size1;
	}
	
	float distance2=0;
	for(int i=0; i<size0; i++){
		distance2 += (pos0[i]-pos1[i])*(pos0[i]-pos1[i]);
	}
	float distance = sqrt(distance2);
	
	float tij_bar = _d_norm * distance/_v_cond+1;
	float tij=0;
	_rnd.gen_normal_cpu(&tij, 1, tij_bar, 0.1*tij_bar);
	int delay = ceil(tij*0.001/_dt);
	return delay;
}

}
}
