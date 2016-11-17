#include "gsbn/Rec.hpp"

namespace gsbn{

Rec::Rec(){
}

void Rec::init(RecParam rec_param, Database& db){
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

void Rec::record(bool force){
	if(!_enable){
		return;
	}

	int simstep = static_cast<const int *>(_conf->cpu_data(0))[Database::IDX_CONF_TIMESTAMP];
	float dt = static_cast<const float *>(_conf->cpu_data(0))[Database::IDX_CONF_DT];
	float prn = static_cast<const float *>(_conf->cpu_data(0))[Database::IDX_CONF_PRN];
	CHECK_GE(simstep, 0);
	CHECK_GE(dt, 0);
	
	if(simstep < _offset){
		return;
	}
		
	if(_spike_period>0 && (simstep%_spike_period)==0){
		string filename1 = _directory+"/spike.csv";
		fstream output1(filename1, ios::out | ios::app);
		
		vector<SyncVector<int>*> spikes;
		int i=0;
		SyncVector<int>* spike;
		while(spike=_db->sync_vector_i("spike_"+to_string(i))){
			i++;
			spikes.push_back(spike);
		}
		int pop_id=0;
		for(vector<SyncVector<int>*>::iterator it=spikes.begin(); it!=spikes.end(); it++){
			output1 << simstep*dt << "," << pop_id;
			int size=(*it)->cpu_vector()->size();
			for(int i=0; i<size; i++){
				if((*((*it)->cpu_vector()))[i]!=0){
					output1 << ","<<i;
				}
			}
			output1<<endl;
			pop_id++;
		}
	}
	
	if((force) || (_snapshot_period>0 && (simstep%_snapshot_period)==0)){
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

}
