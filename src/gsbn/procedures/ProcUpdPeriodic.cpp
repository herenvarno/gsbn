#include "gsbn/procedures/ProcUpdPeriodic.hpp"

namespace gsbn{
namespace proc_upd_periodic{

REGISTERIMPL(ProcUpdPeriodic);

void ProcUpdPeriodic::init_new(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_new(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt);
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
			proj->init_new(proc_param, proj_param, db, &_list_proj, &_list_pop);
		}
	}
}

void ProcUpdPeriodic::init_copy(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_copy(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt);
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
			proj->init_copy(proc_param, proj_param, db, &_list_proj, &_list_pop);
		}
	}
}


void ProcUpdPeriodic::update_cpu(){
	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if(cycle_flag != 1){
		return;
	}
	
	int simstep;
	float dt;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("dt", dt));
	if(simstep%(int(1/dt))==0){
		LOG(INFO) << "Sim [ " << simstep * dt<< " ]";
	}

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
		(*it)->update_siq_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_ij_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_j_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_i_cpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_epsc_cpu();
	}
}

#ifndef CPU_ONLY

void ProcUpdPeriodic::update_gpu(){
	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if(cycle_flag != 1){
		return;
	}
	
	int simstep;
	float dt;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("dt", dt));
	if(simstep%(int(1/dt))==0){
		LOG(INFO) << "Sim [ " << simstep * dt<< " ]";
	}
	
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
		(*it)->update_siq_gpu();
	}
	for(vector<Proj*>::iterator it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_zep_gpu();
	}
	cudaDeviceSynchronize();
}

#endif

}
}
