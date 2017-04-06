#include "gsbn/procedures/ProcMail.hpp"

namespace gsbn{
namespace proc_mail{

REGISTERIMPL(ProcMail);

void ProcMail::init_new(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	_msg.init_new(net_param, db);
	
	CHECK(_glv.getf("dt", _dt));
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	Parser par(proc_param);
	float init_conn_rate=0;
	if(par.argf("init conn rate", init_conn_rate)){
		if(init_conn_rate>1.0){
			init_conn_rate = 1.0;
		}
	}
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
	
	int rank;
	CHECK(_glv.geti("rank", rank));
	
	int pop_id=0;
	int hcu_cnt=0;
	int mcu_cnt=0;
	_spike_buffer_size = 1;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			int hcu_start = hcu_cnt;
			int mcu_start = mcu_cnt;
			_pop_hcu_start.push_back(hcu_start);
			_pop_mcu_start.push_back(mcu_start);
			int dim_hcu = pop_param.hcu_num();
			int dim_mcu = pop_param.mcu_num();
			_pop_dim_hcu.push_back(dim_hcu);
			_pop_dim_mcu.push_back(dim_mcu);
			hcu_cnt += dim_hcu;
			mcu_cnt += (dim_hcu * dim_mcu);
			
			vector<vector<vector<int>>> list_avail_hcu;
			list_avail_hcu.resize(dim_hcu * dim_mcu);
			_pop_avail_hcu.push_back(list_avail_hcu);
			
			SyncVector<int> *fanout = db.create_sync_vector_i32("fanout_"+to_string(pop_id));
			CHECK(fanout);
			fanout->resize(dim_hcu*dim_mcu, pop_param.fanout_num());
			_pop_fanout.push_back(fanout);
			
			_pop_spike.push_back(NULL);
			
			_pop_avail_proj.resize(pop_id+1);
			_pop_avail_proj_hcu_start.resize(pop_id+1);
			
			vector<int> s;
			for(int k=0; k<pop_param.shape_size(); k++){
				s.push_back(pop_param.shape(k));
			}
			_pop_shape.push_back(s);
			
			_pop_rank.push_back(pop_param.rank());
			
			pop_id++;
		}
	}
	int proj_id=0;
	int total_pop_num = _pop_dim_hcu.size();
	int proj_param_size = net_param.proj_param_size();
	_avail_mails.resize(proj_param_size);
	
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			vector<int> list;
			for(int i=0; i<_pop_dim_hcu[dest_pop]; i++){
				list.push_back(i);
			}
			for(int i=0; i<_pop_dim_hcu[src_pop] * _pop_dim_mcu[src_pop]; i++){
				_pop_avail_hcu[src_pop][i].push_back(list);
			}
			_pop_avail_proj[src_pop].push_back(proj_id);
			_pop_avail_proj_hcu_start[src_pop].push_back(_pop_hcu_start[dest_pop]);
			
			if(_pop_rank[dest_pop]==rank){
				_proj_src_pop.push_back(src_pop);
				_proj_dest_pop.push_back(dest_pop);
			}else{
				_proj_src_pop.push_back(-1);
				_proj_dest_pop.push_back(-1);
			}
			
			int dim_hcu = _pop_dim_hcu[dest_pop];
			int dim_mcu = _pop_dim_mcu[dest_pop];
			int dim_conn = (_pop_dim_hcu[src_pop] * _pop_dim_mcu[src_pop]);
			int slot_num = proj_param.slot_num();
			if(dim_conn > slot_num){
				dim_conn = slot_num;
			}
			
			SyncVector<int> *slot=NULL;
			SyncVector<int> *ii = NULL;
			SyncVector<int> *di = NULL;
			if(_pop_rank[dest_pop]==rank){
				slot= db.create_sync_vector_i32("slot_"+to_string(proj_id));
				CHECK(slot);
				slot->resize(dim_hcu*dim_mcu, slot_num);
				ii= db.sync_vector_i32("ii_"+to_string(proj_id));
				CHECK(ii);
				CHECK_EQ(ii->size(), dim_hcu*dim_conn);
				di= db.sync_vector_i32("di_"+to_string(proj_id));
				CHECK(di);
				CHECK_EQ(di->size(), dim_hcu*dim_conn);
			}
			_proj_slot.push_back(slot);
			_proj_ii.push_back(ii);
			_proj_di.push_back(di);
			
			vector<int> conn_cnt(dim_hcu, 0);
			_proj_conn_cnt.push_back(conn_cnt);
			
			_proj_dim_conn.push_back(dim_conn);
			_proj_distance.push_back(proj_param.distance());
			
			if(_pop_spike[src_pop]==NULL && _pop_rank[dest_pop]==rank){
				SyncVector<int8_t> *spike = db.sync_vector_i8("si_"+to_string(proj_id));
				CHECK(spike);
				if(spike->ld()>0){
					if(_spike_buffer_size!=1){
						CHECK_EQ(_spike_buffer_size, spike->size()/spike->ld());
					}else{
						_spike_buffer_size = spike->size()/spike->ld();
					}
				}
				CHECK_EQ(spike->size(), _spike_buffer_size * dim_hcu*dim_mcu);
				_pop_spike[src_pop]=spike;
			}
			
			proj_id++;
		}
	}

	for(int i=0; i<proj_param_size; i++){
		if(_proj_src_pop[i]>=0){
			init_proj_conn(i, init_conn_rate);
		}
	}
}

void ProcMail::init_copy(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	_msg.init_copy(net_param, db);
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	CHECK(_glv.getf("dt", _dt));
	
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
	
	int pop_id=0;
	int hcu_cnt=0;
	int mcu_cnt=0;
	_spike_buffer_size = 1;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			int hcu_start = hcu_cnt;
			int mcu_start = mcu_cnt;
			_pop_hcu_start.push_back(hcu_start);
			_pop_mcu_start.push_back(mcu_start);
			int dim_hcu = pop_param.hcu_num();
			int dim_mcu = pop_param.mcu_num();
			_pop_dim_hcu.push_back(dim_hcu);
			_pop_dim_mcu.push_back(dim_mcu);
			hcu_cnt += dim_hcu;
			mcu_cnt += (dim_hcu * dim_mcu);
			
			vector<vector<vector<int>>> list_avail_hcu;
			list_avail_hcu.resize(dim_hcu * dim_mcu);
			_pop_avail_hcu.push_back(list_avail_hcu);
			
			SyncVector<int> *fanout = db.sync_vector_i32("fanout_"+to_string(pop_id));
			CHECK(fanout);
			CHECK_EQ(fanout->size(), dim_hcu*dim_mcu);
			_pop_fanout.push_back(fanout);
			SyncVector<int8_t> *spike = db.sync_vector_i8("si_"+to_string(pop_id));
			CHECK(spike);
			if(spike->ld()>0){
				if(_spike_buffer_size!=1){
					CHECK_EQ(_spike_buffer_size, spike->size()/spike->ld());
				}else{
					_spike_buffer_size = spike->size()/spike->ld();
				}
			}
			CHECK_EQ(spike->size(), _spike_buffer_size * dim_hcu*dim_mcu);
			_pop_spike.push_back(spike);
			
			_pop_avail_proj.resize(pop_id+1);
			_pop_avail_proj_hcu_start.resize(pop_id+1);
			
			vector<int> s;
			for(int k=0; k<pop_param.shape_size(); k++){
				s.push_back(pop_param.shape(k));
			}
			_pop_shape.push_back(s);
			pop_id++;
		}
	}
	int proj_id=0;
	int total_pop_num = _pop_dim_hcu.size();
	int proj_param_size = net_param.proj_param_size();
	_avail_mails.resize(proj_param_size);
	
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			
			_pop_avail_proj[src_pop].push_back(proj_id);
			_pop_avail_proj_hcu_start[src_pop].push_back(_pop_hcu_start[dest_pop]);
			
			_proj_src_pop.push_back(src_pop);
			_proj_dest_pop.push_back(dest_pop);
			
			int dim_hcu = _pop_dim_hcu[dest_pop];
			int dim_mcu = _pop_dim_mcu[dest_pop];
			int dim_conn = (_pop_dim_hcu[src_pop] * _pop_dim_mcu[src_pop]);
			int slot_num = proj_param.slot_num();
			if(dim_conn > slot_num){
				dim_conn = slot_num;
			}
			
			SyncVector<int> *slot = db.sync_vector_i32("slot_"+to_string(proj_id));
			CHECK(slot);
			CHECK_EQ(slot->size(), dim_hcu*dim_mcu);
			_proj_slot.push_back(slot);
			SyncVector<int> *ii = db.sync_vector_i32("ii_"+to_string(proj_id));
			CHECK(ii);
			CHECK_EQ(ii->size(), dim_hcu*dim_conn);
			_proj_ii.push_back(ii);
			SyncVector<int> *di = db.sync_vector_i32("di_"+to_string(proj_id));
			CHECK(di);
			CHECK_EQ(di->size(), dim_hcu*dim_conn);
			_proj_di.push_back(di);
			
			_proj_dim_conn.push_back(dim_conn);
			_proj_distance.push_back(proj_param.distance());
			
			const int *ptr_ii = ii->cpu_data();
			vector<int> conn_cnt(dim_hcu);
			for(int i=0; i<dim_hcu; i++){
				for(int j=0; j<dim_conn; j++){
					if(ptr_ii[i*dim_conn+j]<0){
						conn_cnt[i]=j;
						break;
					}
					conn_cnt[i]=dim_conn;
				}
			}
			_proj_conn_cnt.push_back(conn_cnt);
			
			vector<int> list;
			for(int i=0; i<_pop_dim_hcu[dest_pop]; i++){
				list.push_back(i);
			}
			for(int i=0; i<_pop_dim_hcu[src_pop] * _pop_dim_mcu[src_pop]; i++){
				vector<int> list_cpy=list;
				for(int x=0; x<dim_hcu; x++){
					for(int y=0; y<conn_cnt[x]; y++){
						int mcu = ptr_ii[x*dim_conn+y];
						if(mcu==i){
							for(int z=0; z<list_cpy.size(); z++){
								if(list_cpy[z]==x){
									list_cpy.erase(list_cpy.begin()+z);
									break;
								}
							}
							break;
						}
					}
				}
				_pop_avail_hcu[src_pop][i].push_back(list_cpy);
			}
			
			proj_id++;
		}
	}
}

void ProcMail::update_cpu(){
	_msg.update();
	receive_spike();
	send_spike();
}

#ifndef CPU_ONLY
void ProcMail::update_gpu(){
	update_cpu();
}
#endif


void ProcMail::send_spike(){
	bool plasticity;
	CHECK(_glv.getb("plasticity", plasticity));
	if(!plasticity){
		return;
	}
	
	// SEND
	int timestep;
	CHECK(_glv.geti("simstep", timestep));
	
	int spike_buffer_cursor = timestep % _spike_buffer_size;
	for(int p=0; p<_pop_dim_hcu.size(); p++){
		if(_pop_spike[p]==NULL){
			continue;
		}
		int *ptr_fanout = _pop_fanout[p]->mutable_cpu_data();
		const int8_t *ptr_spike = _pop_spike[p]->cpu_data()+spike_buffer_cursor*_pop_dim_hcu[p]*_pop_dim_mcu[p];
		for(int i=0; i<_pop_dim_hcu[p] * _pop_dim_mcu[p]; i++){
			int hcu_idx = i/_pop_dim_mcu[p];
			int size=0;
			for(int j=0; j<_pop_avail_hcu[p][i].size(); j++){
				size+=_pop_avail_hcu[p][i][j].size();
			}
			if(ptr_spike[i]<=0 || ptr_fanout[i]<=0 || size<=0){
				continue;
			}
			ptr_fanout[i]--;
			float random_number;
			_rnd.gen_uniform01_cpu(&random_number);
			int dest_hcu_idx = ceil(random_number*size-1);
			for(int j=0; j<_pop_avail_hcu[p][i].size(); j++){
				if(dest_hcu_idx < _pop_avail_hcu[p][i][j].size()){
					int proj = _pop_avail_proj[p][j];
					int src_pop = _proj_src_pop[proj];
					int dest_pop = _proj_dest_pop[proj];
					int src_hcu = hcu_idx;
					int dest_hcu = _pop_avail_hcu[p][i][j][dest_hcu_idx];
					Coordinate c0(src_hcu, _pop_dim_hcu[src_pop] * _pop_dim_mcu[src_pop], _pop_shape[src_pop]);
					Coordinate c1(dest_hcu, _pop_dim_hcu[dest_pop] * _pop_dim_mcu[dest_pop], _pop_shape[dest_pop]);
					int delay = delay_cycle(proj, c0, c1);
					_msg.send(proj, i, _pop_avail_hcu[p][i][j][dest_hcu_idx], 1, delay);
					break;
				}else{
					dest_hcu_idx -= _pop_avail_hcu[p][i][j].size();
				}
			}
		}
	}
}

void ProcMail::receive_spike(){
	// RECEIVE
	bool plasticity;
	CHECK(_glv.getb("plasticity", plasticity));
	if(!plasticity){
		return;
	}
	
	for(int prj=0; prj<_proj_src_pop.size(); prj++){
		if(_proj_src_pop[prj]<0){
			continue;
		}
		vector<msg_t> list_msg = _msg.receive(prj);
		int src_pop = _proj_src_pop[prj];
		int dest_pop = _proj_dest_pop[prj];
		for(vector<msg_t>::iterator it = list_msg.begin(); it!=list_msg.end(); it++){
			if(it->dest_hcu < 0 ||
				it->dest_hcu > _pop_dim_hcu[dest_pop] ||
				it->src_mcu < 0 ||
				it->src_mcu > _pop_dim_hcu[src_pop]*_pop_dim_mcu[src_pop]){
				continue;
			}
			switch(it->type){
			case 1:
				if((*(_proj_slot[prj]->mutable_cpu_vector()))[it->dest_hcu]>0){
					(*(_proj_slot[prj]->mutable_cpu_vector()))[it->dest_hcu]--;
					_msg.send(prj, it->src_mcu, it->dest_hcu, 2, it->delay);
					add_row(it->proj, it->src_mcu, it->dest_hcu, it->delay);
				}else{
					_msg.send(prj, it->src_mcu, it->dest_hcu, 3, it->delay);
				}
				break;
			case 2:
				update_avail_hcu(src_pop, it->src_mcu, prj, it->dest_hcu, true);
				break;
			case 3:
				update_avail_hcu(src_pop, it->src_mcu, prj, it->dest_hcu, false);
				(_pop_fanout[src_pop]->mutable_cpu_data())[it->src_mcu]++;
				break;
			default:
				break;
			}
		}
	}
}

void ProcMail::add_row(int proj, int src_mcu, int dest_hcu, int delay){
	if(_proj_conn_cnt[proj][dest_hcu]<_proj_dim_conn[proj]){
		int idx = _proj_conn_cnt[proj][dest_hcu];
		_proj_ii[proj]->mutable_cpu_data()[dest_hcu*_proj_dim_conn[proj]+idx]=src_mcu;
		_proj_di[proj]->mutable_cpu_data()[dest_hcu*_proj_dim_conn[proj]+idx]=delay;
		_proj_conn_cnt[proj][dest_hcu]++;
	}
}

void ProcMail::update_avail_hcu(int pop, int src_mcu, int proj_id, int dest_hcu, bool remove_all){
	if(src_mcu >= _pop_avail_hcu[pop].size()){
		return;
	}
	
	int hcu_start = -1;
	int idx_proj = 0;
	for(int idx_proj=0; idx_proj<_pop_avail_proj[pop].size(); idx_proj++){
		if(_pop_avail_proj[pop][idx_proj]==proj_id){
			hcu_start = _pop_avail_proj_hcu_start[pop][idx_proj];
			break;
		}
	}
	if(idx_proj >= _pop_avail_proj[pop].size()){
		return;
	}
	
	if(remove_all){
		for(int i=0; i<_pop_avail_proj_hcu_start[pop].size(); i++){
			if(_pop_avail_proj_hcu_start[pop][i]==hcu_start){
				for(int j=0; j<_pop_avail_hcu[pop][src_mcu][i].size(); j++){
					if(_pop_avail_hcu[pop][src_mcu][i][j]==dest_hcu){
						_pop_avail_hcu[pop][src_mcu][i].erase(_pop_avail_hcu[pop][src_mcu][i].begin()+j);
						break;
					}
				}
			}
		}
	}else{
		for(int j=0; j<_pop_avail_hcu[pop][src_mcu][idx_proj].size(); j++){
			if(_pop_avail_hcu[pop][src_mcu][idx_proj][j]==dest_hcu){
				_pop_avail_hcu[pop][src_mcu][idx_proj].erase(_pop_avail_hcu[pop][src_mcu][idx_proj].begin()+j);
				break;
			}
		}
	}
}

bool ProcMail::validate_conn(int pop, int src_mcu, int proj_id, int dest_hcu){
	if(src_mcu >= _pop_avail_hcu[pop].size()){
		return false;
	}
	
	const int *ptr_fanout = _pop_fanout[pop]->cpu_data();
	if(ptr_fanout[src_mcu]<=0){
		return false;
	}
	
	int i=0;
	for(i=0; i<_pop_avail_proj[pop].size(); i++){
		if(_pop_avail_proj[pop][i]==proj_id){
			break;
		}
	}
	if(i>=_pop_avail_proj[pop].size()){
		return false;
	}
	
	for(int j=0; j<_pop_avail_hcu[pop][src_mcu][i].size(); j++){
		if(_pop_avail_hcu[pop][src_mcu][i][j]==dest_hcu){
			return true;
			break;
		}
	}
	
	return false;
}

void ProcMail::init_proj_conn(int proj, int init_conn_rate){
	int src_pop = _proj_src_pop[proj];
	int dest_pop = _proj_dest_pop[proj];
	if(init_conn_rate>0.0){
		int conn_per_hcu = int(init_conn_rate * _proj_dim_conn[proj]);
		for(int i=0; i<_pop_dim_hcu[dest_pop]; i++){
			vector<int> avail_mcu_list(_pop_dim_hcu[src_pop]*_pop_dim_mcu[src_pop]);
			std::iota(std::begin(avail_mcu_list), std::end(avail_mcu_list), 0);
			for(int j=0; j<conn_per_hcu && !avail_mcu_list.empty(); j++){
				while(!avail_mcu_list.empty()){
					float random_number;
					_rnd.gen_uniform01_cpu(&random_number);
					int src_mcu_idx = ceil(random_number*(avail_mcu_list.size())-1);
					int src_mcu = avail_mcu_list[src_mcu_idx];
					if(validate_conn(src_pop, src_mcu, proj, i)){
						int *ptr_fanout = _pop_fanout[src_pop]->mutable_cpu_data();
						ptr_fanout[src_mcu]--;
						update_avail_hcu(src_pop, src_mcu, proj, i, true);
						(*(_proj_slot[proj]->mutable_cpu_vector()))[i]--;
						Coordinate c0(src_mcu/_pop_dim_mcu[src_pop], _pop_dim_hcu[src_pop] * _pop_dim_mcu[src_pop], _pop_shape[src_pop]);
						Coordinate c1(i, _pop_dim_hcu[dest_pop] * _pop_dim_mcu[dest_pop], _pop_shape[dest_pop]);
						int delay = delay_cycle(proj, c0, c1);
						add_row(proj, src_mcu, i, delay);
						avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
						break;
					}
					avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
				}
			}
		}
	}
}

int ProcMail::delay_cycle(int proj, Coordinate c0, Coordinate c1){
	float tij_bar = _d_norm * (c0.distance_to(c1, _proj_distance[proj]))/_v_cond+1;
	float tij=0;
	_rnd.gen_normal_cpu(&tij, 1, tij_bar, 0.1*tij_bar);
	int delay = ceil(tij*0.001/_dt);
	return delay;
}

}
}
