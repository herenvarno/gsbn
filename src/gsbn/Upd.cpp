#include "gsbn/Upd.hpp"

namespace gsbn{


Upd::Upd() : _list_proc() {
}

void Upd::init_new(SolverParam solver_param, Database& db){
	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		string proc_name=solver_param.proc_param(i).name();
		ProcedureBase *proc = ProcedureFactory::create(proc_name);
		_list_proc.push_back(proc);
		proc->init_new(solver_param, db);
	}
}

void Upd::init_copy(SolverParam solver_param, Database& db){
	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		string proc_name=solver_param.proc_param(i).name();
		ProcedureBase *proc = ProcedureFactory::create(proc_name);
		_list_proc.push_back(proc);
		proc->init_copy(solver_param, db);
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
