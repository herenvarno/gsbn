#include "gsbn/Upd.hpp"

namespace gsbn{


Upd::Upd() : _list_proc() {
}

void Upd::init_new(NetParam net_param, Database& db){
	int procedure_size = net_param.procedure_size();
	for(int i=0; i<procedure_size; i++){
		string proc_name=net_param.procedure(i);
		ProcedureBase *proc = ProcedureFactory::create(proc_name);
		_list_proc.push_back(proc);
		proc->init_new(net_param, db);
	}
}

void Upd::init_copy(NetParam net_param, Database& db){
	int procedure_size = net_param.procedure_size();
	for(int i=0; i<procedure_size; i++){
		string proc_name=net_param.procedure(i);
		ProcedureBase *proc = ProcedureFactory::create(proc_name);
		_list_proc.push_back(proc);
		proc->init_copy(net_param, db);
	}
}



void Upd::update(){
	for(vector<ProcedureBase*>::iterator it=_list_proc.begin(); it!=_list_proc.end(); it++){
		if(mode()!=GPU){
			LOG(INFO) << "update!!";
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
