#include "gsbn/procedures/ProcSpkRec.hpp"

namespace gsbn{
namespace proc_spk_rec{

REGISTERIMPL(ProcSpkRec);

void ProcSpkRec::init_new(SolverParam solver_param, Database& db){
	_db = &db;
	
	string log_dir;
	CHECK(_glv.gets("log-dir", log_dir));
	CHECK(!log_dir.empty());
	
	string dir = log_dir + __PROC_NAME__;
	
	struct stat info;
	/* Check directory exists */
	if( stat( dir.c_str(), &info ) == 0 && (info.st_mode & S_IFDIR)){
		_directory = dir;
	} else {
		LOG(WARNING) << "Directory does not exist! Create one!";
		string cmd="mkdir -p "+dir;
		if(system(cmd.c_str())!=0){
			LOG(FATAL) << "Cannot create directory for state records! Aboart!";
		}
		_directory = dir;
	}
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	Parser par(proc_param);
	if(!par.argi("offset", _offset)){
		_offset = 0;
	}else{
		CHECK_GE(_offset,0);
	}
	if(!par.argi("period", _period)){
		_period = 1;
	}else{
		CHECK_GT(_period, 0);
	}
	
	int i=0;
	_spike_buffer_size = 1;
	SyncVector<int8_t>* spike;
	while(spike=_db->sync_vector_i8("spike_"+to_string(i))){
		_spikes.push_back(spike);
		if(spike->ld()>0){
			if(_spike_buffer_size!=1){
				CHECK_EQ(_spike_buffer_size, spike->size()/spike->ld());
			}else{
				_spike_buffer_size = spike->size()/spike->ld();
			}
		}
		string filename = _directory+"/spk_pop_"+ to_string(i) +".csv";
		fstream output(filename, ios::out| std::ofstream::trunc);
		output.close();
		i++;
	}
}

void ProcSpkRec::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcSpkRec::update_cpu(){

	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if(cycle_flag != 1){
		return;
	}
	
	int simstep;
	float dt;
	float prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK_GE(simstep, 0);
	
	if(simstep < _offset){
		return;
	}
		
	if((simstep%_period)==0){
		int cursor = 0;
		if(_spike_buffer_size > 1){
			cursor = simstep % _spike_buffer_size;
		}
		int pop_id=0;
		for(vector<SyncVector<int8_t>*>::iterator it=_spikes.begin(); it!=_spikes.end(); it++){
			string filename = _directory+"/spk_pop_"+ to_string(pop_id) +".csv";
			fstream output(filename, ios::out | ios::app);
			bool flag=false;
			int size = (*it)->size();
			if(_spike_buffer_size>1){
				size=(*it)->ld();
			}
			for(int i=0; i<size; i++){
				int8_t spike_block=((*it)->cpu_data(cursor))[i];
				if(spike_block>0){
					if(!flag){
						output << simstep;
						flag = true;
					}
					output << ","<<i;
				}
			}
			if(flag){
				output<<endl;
			}
			pop_id++;
		}
	}
}

#ifndef CPU_ONLY
void ProcSpkRec::update_gpu(){
	update_cpu();
}
#endif
}
}
