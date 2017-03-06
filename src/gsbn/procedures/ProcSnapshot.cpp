#include "gsbn/procedures/ProcSnapshot.hpp"

namespace gsbn{
namespace proc_snapshot{

REGISTERIMPL(ProcSnapshot);

void ProcSnapshot::init_new(SolverParam solver_param, Database& db){
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
}

void ProcSnapshot::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcSnapshot::update_cpu(){
	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if(cycle_flag != 1){
		return;
	}
	
	int simstep;
	float dt;
	float prn;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("dt", dt));
	CHECK(_glv.getf("prn", prn));
	CHECK_GE(simstep, 0);
	CHECK_GE(dt, 0);
	
	if(simstep < _offset){
		return;
	}
	
	if((simstep%_period)==0){
		SolverState st = _db->state_to_proto();
		st.set_timestamp(simstep*dt);
		st.set_prn(prn);

		string filename = _directory+"/Snapshot_"+to_string(simstep*dt)+".bin";
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
