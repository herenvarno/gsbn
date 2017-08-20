#include "gsbn/Upd.hpp"

namespace gsbn{


Upd::Upd() : _list_proc() {
}

Upd::~Upd(){
}

void Upd::init(SolverParam solver_param, Database& db, bool initialized_db){
	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		string proc_name=solver_param.proc_param(i).name();
		ProcedureBase *proc = ProcedureFactory::create(proc_name);
		_list_proc.push_back(proc);
		LOG(INFO) << "Init procedure " << proc_name;
		if(!initialized_db){
			proc->init_new(solver_param, db);
		}else{
			proc->init_copy(solver_param, db);
		}
		
	}
}

void Upd::update(){
	for(vector<ProcedureBase*>::iterator it=_list_proc.begin(); it!=_list_proc.end(); it++){
		if(mode()!=GPU){
			(*it)->update_cpu();
		}else{
			#ifndef CPU_ONLY
			(*it)->update_gpu();
			#else
			__NO_GPU__;
			#endif
		}
	}
}

}
