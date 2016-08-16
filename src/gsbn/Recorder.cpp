#include "gsbn/Recorder.hpp"

namespace gsbn{

void Recorder::set_directory(string dir){
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

void Recorder::set_timestamp(int timestamp){
	CHECK_GE(timestamp, 0);
	_timestamp = timestamp;
}

void Recorder::set_freq(int freq){
	CHECK_GT(_freq, 0);
	_freq = freq;
}


void Recorder::append_table(Table tab){
	_tables.push_back(tab);
}

void Recorder::record(){
	CHECK(!_directory.empty());
	
	if(_timestamp%_freq!=0){
		return;
	}
	
	SolverState st;
	st.set_timestamp((unsigned int)_timestamp);
	for (std::vector<Table>::const_iterator iterator = _tables.begin(),
		end = _tables.end(); iterator != end; ++iterator) {
		LOG(INFO) << "1";
		TableState *tab_st = st.add_table_state();
		LOG(INFO) << "2";
		Table tab=*iterator;
		LOG(INFO) << "3";
		*tab_st = tab.state();
		LOG(INFO) << "4";
	}
	
	string filename = _directory+"/SolverState_"+to_string(_timestamp)+".bin";
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	if (!st.SerializeToOstream(&output)) {
		LOG(FATAL) << "Failed to write states.";
	}
	
}

}
