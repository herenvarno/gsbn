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
	
	CHECK(_glv.geti("rank", _rank));
	float dt;
	CHECK(_glv.getf("dt", dt));
	
	NetParam net_param = solver_param.net_param();
	
	int pop_id=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop p(pop_id, pop_param, db);
			if(p._rank==_rank){
				_pop_list.push_back(p);
			}
		}
	}
	
	for(int i=0; i<_pop_list.size(); i++){
		Pop p=_pop_list[i];
		string filename = _directory+"/spk_pop_"+ to_string(p._id) +".csv";
		fstream output(filename, ios::out| std::ofstream::trunc);
		output << p._dim_hcu << "," << p._dim_mcu << "," << dt << "," << p._maxfq << endl;
		output.close();
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
	CHECK(_glv.geti("simstep", simstep));
	CHECK_GE(simstep, 0);
	
	if(simstep < _offset){
		return;
	}
		
	if((simstep%_period)==0){
		for(int i=0; i<_pop_list.size(); i++){
			Pop p=_pop_list[i];
			string filename = _directory+"/spk_pop_"+ to_string(p._id) +".csv";
			p.record(filename, simstep);
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
