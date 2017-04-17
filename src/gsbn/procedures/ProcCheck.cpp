#include "gsbn/procedures/ProcCheck.hpp"

namespace gsbn{
namespace proc_check{

REGISTERIMPL(ProcCheck);

void ProcCheck::init_new(SolverParam solver_param, Database& db){

	string log_dir;
	CHECK(_glv.gets("log-dir", log_dir));
	CHECK(!log_dir.empty());
	string dir = log_dir + __PROC_NAME__;
	struct stat info;
	/* Check directory exists */
	if( !(stat( dir.c_str(), &info ) == 0 && (info.st_mode & S_IFDIR))){
		LOG(WARNING) << "Directory does not exist! Create one!";
		string cmd="mkdir -p "+dir;
		if(system(cmd.c_str())!=0){
			LOG(FATAL) << "Cannot create directory for state records! Aboart!";
		}
	}
	_logfile = dir + "/check_result.txt";
	
	_db = &db;
	
	GenParam gen_param = solver_param.gen_param();
	
	float dt = gen_param.dt();
	
	// mode
	int mode_param_size = gen_param.mode_param_size();
	int max_step=-1;
	for(int i=0;i<mode_param_size;i++){
		ModeParam mode_param=gen_param.mode_param(i);
		
		int begin_step = int(mode_param.begin_time()/dt);
		int end_step = int(mode_param.end_time()/dt);
		CHECK_GE(begin_step, max_step)
			<< "Order of modes is wrong or there is overlapping time range, abort!";
		CHECK_GE(end_step, begin_step)
			<< "Time range is wrong, abort!";
	
		int begin_lgidx_id = mode_param.begin_lgidx_id();
		int begin_lgexp_id = mode_param.begin_lgexp_id();
		int begin_wmask_id = mode_param.begin_wmask_id();
		int time_step = mode_param.time_step();
		int lgidx_step = mode_param.lgidx_step();
		int lgexp_step = mode_param.lgexp_step();
		int wmask_step = mode_param.wmask_step();
		float prn = mode_param.prn();
		if(prn!=0.0)
			continue;
		int plasticity = mode_param.plasticity();
		while(begin_step<end_step){
			mode_t m;
			m.begin_step = begin_step;
			m.end_step = begin_step+time_step;
			m.lgidx_id = begin_lgidx_id;
			m.lgexp_id = begin_lgexp_id;
			m.wmask_id = begin_wmask_id;
			begin_lgidx_id+=lgidx_step;
			begin_lgexp_id+=lgexp_step;
			begin_wmask_id+=wmask_step;
			m.prn = prn;
			m.plasticity = plasticity;
			_list_mode.push_back(m);
			begin_step+=time_step;
		}
	}
	
	CHECK(_lgidx = db.sync_vector_i32(".lgidx"));
	CHECK(_lginp = db.sync_vector_f32(".lginp"));
	CHECK(_wmask = db.sync_vector_f32(".wmask"));
	
	int rank;
	CHECK(_glv.geti("rank", rank));
	
	int pop_num_acc=0;
	int pop_hcu_acc=0;
	int pop_mcu_acc=0;
	
	NetParam net_param=solver_param.net_param();
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		for(int j=0; j<pop_param.pop_num(); j++){
			Pop p(pop_num_acc, pop_hcu_acc, pop_mcu_acc, pop_param, db);
			if(p._rank==rank){
				_pop_list.push_back(p);
			}
		}
	}
	
	_total_hcu_num = pop_hcu_acc;
	if(rank==0){
		_shared_idx.resize(pop_hcu_acc);
		_shared_cnt.resize(pop_hcu_acc);
	}
	
	_cursor = 0;
	_pattern_num=0;
	_correct_pattern_num=0;
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	Parser par(proc_param);
	if(!par.argi("threashold", _threashold)){
		_threashold = 0;
	}
	
	_spike_buffer_size = 1;
	for(int i=0; i<_pop_list.size(); i++){
		SyncVector<int8_t>* spike = _pop_list[i]._spike;
		if(spike->ld()>0){
			if(_spike_buffer_size!=1){
				CHECK_EQ(_spike_buffer_size, spike->size()/spike->ld());
			}else{
				_spike_buffer_size = spike->size()/spike->ld();
			}
		}
	}
	
	fstream output(_logfile, ios::out| std::ofstream::trunc);
	output<<"result | reference pattern | recorded pattern | recorded pattern spike count"<< endl;
	output.close();
	
	_updated_flag=false;
	
	if(rank==0){
		MPI_Win_create(&_shared_idx[0], _shared_idx.size(), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &_win_idx);
		MPI_Win_create(&_shared_cnt[0], _shared_cnt.size(), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &_win_cnt);
	}else{
		MPI_Win_create(MPI_BOTTOM, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &_win_idx);
		MPI_Win_create(MPI_BOTTOM, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &_win_cnt);
	}
	
	for(int i=0; i<_pop_list.size(); i++){
		_pop_list[i].set_win_idx(_win_idx);
		_pop_list[i].set_win_cnt(_win_cnt);
	}
	
}

void ProcCheck::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcCheck::update_cpu(){
	
	
	// update cursor
	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	int rank;
	CHECK(_glv.geti("rank", rank));
	
	if(cycle_flag==0){
		return;
	}else if(cycle_flag>0){
		int timestep;
		CHECK(_glv.geti("simstep", timestep));
		
		int spike_buffer_cursor = timestep % _spike_buffer_size;
		int begin_step = _list_mode[_cursor].begin_step;
		int end_step = _list_mode[_cursor].end_step;
		if(timestep <= begin_step){
			return;
		}else if(timestep>begin_step && timestep<end_step){;
			// update counters
			for(int i=0; i<_pop_list.size(); i++){
				_pop_list[i].update_counter(spike_buffer_cursor);
			}
			_updated_flag = true;
			return;
		}else if(timestep>=end_step){
			if(!_updated_flag){
				while(timestep>=_list_mode[_cursor].end_step){
					_cursor++;
				}
				return;
			}
			// calculate count result
			MPI_Win_fence(0, _win_idx);
			MPI_Win_fence(0, _win_cnt);
			for(int i=0; i<_pop_list.size(); i++){
				_pop_list[i].check_result();
				_pop_list[i].send_result();
			}
			MPI_Win_fence(0, _win_idx);
			MPI_Win_fence(0, _win_cnt);
			
			if(rank==0){
				const int *ptr_lgidx = _lgidx->cpu_data(_list_mode[_cursor].lgexp_id);
				bool flag = true;
				for(int i=0; i<_total_hcu_num; i++){
					if(((_shared_cnt[i] < _threashold || _shared_idx[i] !=  ptr_lgidx[i]) && ptr_lgidx[i]>=0)|| (_shared_cnt[i] > _threashold && ptr_lgidx[i]<0)){
						flag=false;
						break;
					}
				}
				
				fstream output(_logfile, ios::out | ios::app);
				output << flag << "|[";
				for(int i=0; i<_total_hcu_num;i++){
					output << ptr_lgidx[i] <<",";
				}
				output << "]|[";
				for(int i=0; i<_shared_idx.size();i++){
					output << _shared_idx[i] <<",";
				}
				output << "]|[";
				for(int i=0; i<_shared_cnt.size();i++){
					output << _shared_cnt[i] <<",";
				}
				output << "]"<<endl;
				
				if(flag==false){
					_pattern_num++;
				}else{
					_pattern_num++;
					_correct_pattern_num++;
				}
			}
			_cursor++;
		}
	}else{
		if(rank == 0){
			cout << "correct pattern: " << _correct_pattern_num << "/" << _pattern_num << "(" << _correct_pattern_num*100.0/float(_pattern_num)<< "%)"<< endl;
			fstream output(_logfile, ios::out | ios::app);
			output << "correct pattern: " << _correct_pattern_num << "/" << _pattern_num << "(" << _correct_pattern_num*100.0/float(_pattern_num)<< "%)"<< endl;
		}
	}
}

#ifndef CPU_ONLY
void ProcCheck::update_gpu(){
	update_cpu();
}
#endif
}
}
