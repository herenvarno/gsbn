#include "gsbn/Rec.hpp"

namespace gsbn{

Rec::Rec(): _directory(), _period(1), _db(NULL), _conf(NULL){
}

void Rec::init(Database& db){
	CHECK(_conf = db.table(".conf"));
	_db = &db;
}


void Rec::set_directory(string dir){
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
}

void Rec::set_period(int period){
	_period = period;
}

void Rec::record(bool force){
	CHECK(!_directory.empty());
	
	if((!force) && (_period<=0)){
//	if((_period<=0)){
                return;
        }
	int simstep = static_cast<const int *>(_conf->cpu_data(0))[Database::IDX_CONF_TIMESTAMP];
	float dt = static_cast<const float *>(_conf->cpu_data(0))[Database::IDX_CONF_DT];
	float prn = static_cast<const float *>(_conf->cpu_data(0))[Database::IDX_CONF_PRN];
	CHECK_GE(simstep, 0);
	CHECK_GE(dt, 0);
	
	string filename1 = _directory+"/SolverState.txt";
	fstream output1(filename1, ios::out | ios::app);
	output1 << simstep*dt;
	SyncVector<int>* spike=_db->sync_vector_i("spike");
	int size=spike->cpu_vector()->size();
	for(int i=0; i<size; i++){
		if((*(spike->cpu_vector()))[i]!=0){
			output1 << ","<<i;
		}
	}
	output1<<endl;
	
	
	if((!force) && simstep%_period!=0){
		return;
	}
	
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
