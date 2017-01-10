#include "gsbn/procedures/ProcFixMix32.hpp"

namespace gsbn{
namespace proc_fix_mix32{

REGISTERIMPL(ProcFixMix32);

void ProcFixMix32::init_new(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	_msg.init_new(net_param, db);
	
	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		ProcParam proc_param=solver_param.proc_param(i);
		if(proc_param.name()=="ProcFixMix32"){
			_norm_frac_bit = proc_param.argi(0);
			_p_frac_bit = proc_param.argi(1);
			CHECK_GE(_norm_frac_bit, 0);
			CHECK_GE(_p_frac_bit, 0);
			break;
		}
	}
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_new(solver_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, &_msg, _norm_frac_bit, _p_frac_bit);
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
			proj->init_new(solver_param, proj_param, db, &_list_proj, &_list_pop, &_msg, _norm_frac_bit, _p_frac_bit);
		}
	}
	
	CHECK(_conf=db.table(".conf"));
}

void ProcFixMix32::init_copy(SolverParam solver_param, Database& db){

	NetParam net_param = solver_param.net_param();
	_msg.init_copy(net_param, db);

	int proc_param_size = solver_param.proc_param_size();
	for(int i=0; i<proc_param_size; i++){
		ProcParam proc_param=solver_param.proc_param(i);
		if(proc_param.name()=="ProcFixMix32"){
			_norm_frac_bit = proc_param.argi(0);
			_p_frac_bit = proc_param.argi(1);
			CHECK_LE(_norm_frac_bit, 31);
			CHECK_LE(_p_frac_bit, 31);
			CHECK_GE(_norm_frac_bit, 0);
			CHECK_GE(_p_frac_bit, 0);
			break;
		}
	}
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_copy(pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, &_msg, _norm_frac_bit, _p_frac_bit);
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
			proj->init_copy(proj_param, db, &_list_proj, &_list_pop, &_msg, _norm_frac_bit, _p_frac_bit);
		}
	}
	
	CHECK(_conf=db.table(".conf"));
}


void ProcFixMix32::update_cpu(){
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
		(*it)->set_from_ext_spike_cpu();
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
		(*it)->receive();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->send();
	}
}

#ifndef CPU_ONLY

void ProcFixMix32::update_gpu(){
	const int* ptr_conf0 = static_cast<const int*>(_conf->cpu_data(0));
	const float* ptr_conf1 = static_cast<const float*>(_conf->cpu_data(0));
	int mode = ptr_conf0[Database::IDX_CONF_MODE];
	if(mode != 1){
		return;
	}
	
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float dt = ptr_conf1[Database::IDX_CONF_DT];
	LOG(INFO) << "Sim [ " << simstep * dt<< " ]";
	
	_msg.update();
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_rnd_gpu();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_sup_gpu();
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
		(*it)->receive();
	}
	for(vector<Pop*>::iterator it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->send();
	}
}

#endif

}
}
