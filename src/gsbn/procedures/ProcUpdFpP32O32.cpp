#include "gsbn/procedures/ProcUpdFpP32O32.hpp"

namespace gsbn{
namespace proc_upd_fp_p32_o32{

REGISTERIMPL(ProcUpdFpP32O32);

void ProcUpdFpP32O32::init_new(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	_msg.init_new(net_param, db);
	
	ProcParam proc_param;
	bool flag=false;
	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		proc_param=solver_param.proc_param(i);
		if(proc_param.name()=="ProcUpdFpP32O32"){
			flag=true;
			break;
		}
	}
	if(!flag){
		LOG(FATAL) << "Can't find the procedure parameter, abort!";
	}
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_new(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, &_msg);
		}
	}
	int total_pop_num = _list_pop.size();
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			Proj *proj = new Proj();
			proj->init_new(proc_param, proj_param, db, &_list_proj, &_list_pop, &_msg);
		}
	}
	CHECK(_conf=db.table(".conf"));
}

void ProcUpdFpP32O32::init_copy(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	_msg.init_copy(net_param, db);
	
	ProcParam proc_param;
	bool flag=false;
	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		proc_param=solver_param.proc_param(i);
		if(proc_param.name()=="ProcUpdFpP32O32"){
			flag=true;
			break;
		}
	}
	if(!flag){
		LOG(FATAL) << "Can't find the procedure parameter, abort!";
	}
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_copy(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, &_msg);
		}
	}
	
	int total_pop_num = _list_pop.size();
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			Proj *proj = new Proj();
			proj->init_copy(proc_param, proj_param, db, &_list_proj, &_list_pop, &_msg);
		}
	}
	
	CHECK(_conf=db.table(".conf"));
}


void ProcUpdFpP32O32::update_cpu(){
	const int* ptr_conf0 = static_cast<const int*>(_conf->cpu_data(0));
	const float* ptr_conf1 = static_cast<const float*>(_conf->cpu_data(0));
	int mode = ptr_conf0[Database::IDX_CONF_MODE];
	if(mode != 1){
		return;
	}
	
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float dt = ptr_conf1[Database::IDX_CONF_DT];
	if(simstep%(int(1/dt))==0){
		LOG(INFO) << "Sim [ " << simstep * dt<< " ]";
	}
	
	_msg.update();
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_rnd_cpu();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_sup_cpu();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->fill_spike();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_full_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_j_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_ss_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_row_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_col_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->receive_spike();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->send_spike();
	}
}

#ifndef CPU_ONLY

void ProcUpdFpP32O32::update_gpu(){
	const int* ptr_conf0 = static_cast<const int*>(_conf->cpu_data(0));
	const float* ptr_conf1 = static_cast<const float*>(_conf->cpu_data(0));
	int mode = ptr_conf0[Database::IDX_CONF_MODE];
	if(mode != 1){
		return;
	}
	
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float dt = ptr_conf1[Database::IDX_CONF_DT];
	if(simstep%(int(1/dt))==0){
		LOG(INFO) << "Sim [ " << simstep * dt<< " ]";
	}
	
	_msg.update();
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_rnd_gpu();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_sup_gpu();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->fill_spike();
	}
	cudaDeviceSynchronize();
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_full_gpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_j_gpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_ss_gpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_row_gpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_col_gpu();
	}
	cudaDeviceSynchronize();
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->receive_spike();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->send_spike();
	}
}

#endif

}
}
