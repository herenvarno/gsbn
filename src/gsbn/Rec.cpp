#include "gsbn/Rec.hpp"

namespace gsbn{

Rec::Rec(): _directory(), _period(1), _tables(), _conf(NULL){
}

void Rec::init(Database& db){
	CHECK(_conf = db.table("conf"));
	_tables = db.tables();
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
	CHECK_GT(_period, 0);
	_period = period;
}

void Rec::record(bool force){
	CHECK(!_directory.empty());
	
	float timestamp = static_cast<const float *>(_conf->cpu_data(0))[Database::IDX_CONF_TIMESTAMP];
	float dt = static_cast<const float *>(_conf->cpu_data(0))[Database::IDX_CONF_DT];
	CHECK_GE(timestamp, 0);
	CHECK_GE(dt, 0);
	
	if((!force) && (int(timestamp/dt)%_period!=0)){
		return;
	}
	
	SolverState st;
	st.set_timestamp(timestamp);
	for (std::vector<Table*>::const_iterator iterator = _tables.begin(),
		end = _tables.end(); iterator != end; ++iterator) {
		TableState *tab_st = st.add_table_state();
		Table *tab=*iterator;
		*tab_st = tab->state();
	}
	
	string filename = _directory+"/SolverState_"+to_string(timestamp)+".bin";
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	if (!st.SerializeToOstream(&output)) {
		LOG(FATAL) << "Failed to write states.";
	}
	
}

}
