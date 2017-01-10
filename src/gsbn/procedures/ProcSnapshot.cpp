#include "gsbn/procedures/ProcSnapshot.hpp"

namespace gsbn{
namespace proc_snapshot{

REGISTERIMPL(ProcSnapshot);

void ProcSnapshot::init_new(SolverParam solver_param, Database& db){

	RecParam rec_param = solver_param.rec_param();
	
	CHECK(_conf = db.table(".conf"));
	_db = &db;
	
	string dir = rec_param.directory();
	CHECK(!dir.empty());
	
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
	
	_enable = rec_param.enable();
	_offset = rec_param.offset();
	_snapshot_period = rec_param.snapshot_period();
	_spike_period = rec_param.spike_period();
	
	if(_spike_period>0){
		string filename1 = _directory+"/spike.csv";
		fstream output1(filename1, ios::out| std::ofstream::trunc);
		output1.close();
	}
}

void ProcSnapshot::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcSnapshot::update_cpu(){
	if(!_enable){
		return;
	}
	
	const int* ptr_conf0 = static_cast<const int*>(_conf->cpu_data(0));
	const float* ptr_conf1 = static_cast<const float*>(_conf->cpu_data(0));
	int mode = ptr_conf0[Database::IDX_CONF_MODE];
	if(mode != 1){
		return;
	}

	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float dt = ptr_conf1[Database::IDX_CONF_DT];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	CHECK_GE(simstep, 0);
	CHECK_GE(dt, 0);
	
	if(simstep < _offset){
		return;
	}
		
	if(_spike_period>0 && (simstep%_spike_period)==0){
		string filename1 = _directory+"/spike.csv";
		fstream output1(filename1, ios::out | ios::app);
		
		vector<SyncVector<int8_t>*> spikes;
		int i=0;
		SyncVector<int8_t>* spike;
		while(spike=_db->sync_vector_i8("spike_"+to_string(i))){
			i++;
			spikes.push_back(spike);
		}
		int pop_id=0;
		for(vector<SyncVector<int8_t>*>::iterator it=spikes.begin(); it!=spikes.end(); it++){
			output1 << simstep << "," << pop_id;
			int size=(*it)->cpu_vector()->size();
			for(int i=0; i<size; i++){
				int8_t spike_block=(*((*it)->cpu_vector()))[i];
				if(spike_block>0){
					output1 << ","<<i;
				}
			}
			output1<<endl;
			pop_id++;
		}
	}
	
	if((_snapshot_period>0 && (simstep%_snapshot_period)==0)){
		SolverState st = _db->state_to_proto();
		st.set_timestamp(simstep*dt);
		st.set_prn(prn);

		string filename = _directory+"/SolverState_"+to_string(simstep*dt)+".bin";
		fstream output(filename, ios::out | ios::trunc | ios::binary);
		if (!st.SerializeToOstream(&output)) {
			LOG(FATAL) << "Failed to write states.";
		}
	}
}

#ifndef CPU_ONLY
void ProcSnapshot::update_gpu(){
	update_cpu();
}
#endif
}
}
