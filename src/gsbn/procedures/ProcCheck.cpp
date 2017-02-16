#include "gsbn/procedures/ProcCheck.hpp"

namespace gsbn{
namespace proc_check{

REGISTERIMPL(ProcCheck);

void ProcCheck::init_new(SolverParam solver_param, Database& db){
	_db = &db;

	GenParam gen_param = solver_param.gen_param();
	
	CHECK(_conf = db.table(".conf"));
	
	// conf
	float* ptr_conf = static_cast<float*>(_conf->mutable_cpu_data(0));
	ptr_conf[Database::IDX_CONF_DT] = gen_param.dt();
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
	CHECK(_count = db.create_sync_vector_i32(".count"));
	
	int mcu_num=0;
	NetParam net_param=solver_param.net_param();
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		mcu_num+=(pop_param.hcu_num()*pop_param.mcu_num());
		for(int j=0; j<pop_param.hcu_num(); j++){
			_mcu_in_hcu.push_back(pop_param.mcu_num());
		}
		_mcu_in_pop.push_back(pop_param.mcu_num()*pop_param.hcu_num());
	}
	_count->resize(mcu_num);
	
	_cursor = 0;
	_pattern_num=0;
	_correct_pattern_num=0;
	
	ProcParam proc_param;
	bool flag=false;
	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		proc_param=solver_param.proc_param(i);
		if(proc_param.name()=="ProcCheck"){
			flag=true;
			break;
		}
	}
	if(flag == false){
		LOG(FATAL) << "No parameters specified for ProcCheck!";
	}
	
	Parser par(proc_param);
	if(!par.argi("threashold", _threashold)){
		_threashold = 0;
	}
	if(!par.args("logfile", _logfile)){
		LOG(FATAL) << "No log file specified for ProcCheck!";
	}
	if(!par.argi("spike buffer size", _spike_buffer_size)){
		_spike_buffer_size = 1;
	}
	_spike_buffer_cursor = 0;
	
	fstream output(_logfile, ios::out| std::ofstream::trunc);
	output<<"result | reference pattern | recorded pattern | recorded pattern spike count"<< endl;
	output.close();
	
}

void ProcCheck::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcCheck::update_cpu(){
	// update cursor
	const int* ptr_conf0 = static_cast<const int*>(_conf->cpu_data(0));
	const float* ptr_conf1 = static_cast<const float*>(_conf->cpu_data(0));
	int mode = ptr_conf0[Database::IDX_CONF_MODE];
	if(mode==0){
		return;
	}else if(mode>0){
		_spike_buffer_cursor++;
		_spike_buffer_cursor %= _spike_buffer_size;
		int timestep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
		int begin_step = _list_mode[_cursor].begin_step;
		int end_step = _list_mode[_cursor].end_step;
		if(timestep <= begin_step){
			return;
		}
		else if(timestep>begin_step && timestep<end_step){
			// update counters
			int *ptr_count = _count->mutable_cpu_data();
			for(int i=0; i<_mcu_in_pop.size(); i++){
				SyncVector<int8_t>* spike;
				CHECK(spike=_db->sync_vector_i8("spike_"+to_string(i)));
				for(int j=0; j<_mcu_in_pop[i]; j++){
					int8_t spike_block=(*(spike->cpu_vector()))[_spike_buffer_cursor*_mcu_in_pop[i]+j];
					if(spike_block>0){
						(*ptr_count)++;
					}
					ptr_count++;
				}
			}
		}else if(timestep>=end_step){
			// calculate count result
			bool flag=true;
			const int *ptr_cnt=_count->cpu_data();
			vector<int> maxcnt_list(_mcu_in_hcu.size(), 0);
			vector<int> maxoff_list(_mcu_in_hcu.size(), -1);
			
			for(int i=0; i<_mcu_in_hcu.size(); i++){
				int maxcnt=0;
				int maxoffset=-1;
				for(int j=0; j<_mcu_in_hcu[i]; j++){
					if(*ptr_cnt>maxcnt){
						maxcnt=*ptr_cnt;
						maxoffset=j;
					}
					ptr_cnt++;
				}
				if(((maxcnt < _threashold || maxoffset != *(_lgidx->cpu_data(_list_mode[_cursor].lgexp_id)+i)) && *(_lgidx->cpu_data(_list_mode[_cursor].lgexp_id)+i)>=0)|| (maxcnt > _threashold && *(_lgidx->cpu_data(_list_mode[_cursor].lgexp_id)+i)<0)){
					flag=false;
					//break;
				}
				maxcnt_list[i]=maxcnt;
				maxoff_list[i]=maxoffset;
			}
			
			fstream output(_logfile, ios::out | ios::app);
			output << flag << "|[";
			for(int i=0; i<_mcu_in_hcu.size();i++){
				output << *(_lgidx->cpu_data(_list_mode[_cursor].lgexp_id)+i) <<",";
			}
			output << "]|[";
			for(int i=0; i<maxcnt_list.size();i++){
				output << maxoff_list[i] <<",";
			}
			output << "]|[";
			for(int i=0; i<maxcnt_list.size();i++){
				output << maxcnt_list[i] <<",";
			}
			output << "]"<<endl;
			
			if(flag==false){
				_pattern_num++;
			}else{
				_pattern_num++;
				_correct_pattern_num++;
			}

			// clear counters
			std::fill(_count->mutable_cpu_vector()->begin(), _count->mutable_cpu_vector()->end(), 0);
			_cursor++;
		}
	}else{
			cout << "correct pattern: " << _correct_pattern_num << "/" << _pattern_num << "(" << _correct_pattern_num*100.0/float(_pattern_num)<< "%)"<< endl;
			fstream output(_logfile, ios::out | ios::app);
			output << "correct pattern: " << _correct_pattern_num << "/" << _pattern_num << "(" << _correct_pattern_num*100.0/float(_pattern_num)<< "%)"<< endl;
	}
}

#ifndef CPU_ONLY
void ProcCheck::update_gpu(){
	update_cpu();
}
#endif
}
}
