#include "gsbn/procedures/ProcUpdMulti.hpp"

namespace gsbn{
namespace proc_upd_multi{

REGISTERIMPL(ProcUpdMulti);

void ProcUpdMulti::init_new(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	#ifndef CPU_ONLY
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	int gpu_device_id=0;
	#endif
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			#ifndef CPU_ONLY
			if(device_count>0){
				gpu_device_id%=device_count;
				p->init_new(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, gpu_device_id+1);
				gpu_device_id++;
			}else{
				p->init_new(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, 0);
			}
			#else
			p->init_new(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, 0);
			#endif
		}
	}
	
	#ifndef CPU_ONLY
	gpu_device_id=0;
	#endif
	
	int total_pop_num = _list_pop.size();
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			Proj *proj = new Proj();
			#ifndef CPU_ONLY
			if(device_count>0){
				gpu_device_id%=device_count;
				proj->init_new(proc_param, proj_param, db, &_list_proj, &_list_pop, gpu_device_id+1);
				gpu_device_id++;
			}else{
				proj->init_new(proc_param, proj_param, db, &_list_proj, &_list_pop, 0);
			}
			#else
			proj->init_new(proc_param, proj_param, db, &_list_proj, &_list_pop, 0);
			#endif
		}
	}
	
}

void ProcUpdMulti::init_copy(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	#ifndef CPU_ONLY
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	int gpu_device_id=0;
	#endif
	
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			#ifndef CPU_ONLY
			if(device_count>0){
				gpu_device_id%=device_count;
				p->init_copy(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, gpu_device_id+1);
				gpu_device_id++;
			}else{
				p->init_copy(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, 0);
			}
			#else
			p->init_copy(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt, 0);
			#endif
		}
	}
	
	#ifndef CPU_ONLY
	gpu_device_id=0;
	#endif
	
	int total_pop_num = _list_pop.size();
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			Proj *proj = new Proj();
			#ifndef CPU_ONLY
			if(device_count>0){
				gpu_device_id%=device_count;
				proj->init_copy(proc_param, proj_param, db, &_list_proj, &_list_pop, gpu_device_id+1);
				gpu_device_id++;
			}else{
				proj->init_copy(proc_param, proj_param, db, &_list_proj, &_list_pop, 0);
			}
			#else
			proj->init_copy(proc_param, proj_param, db, &_list_proj, &_list_pop, 0);
			#endif
		}
	}
	
}


void* th_proj_update_cpu_callback(void *arg){
	Proj* proj=(Proj*)(arg);
	proj->update_cpu();
	return NULL;
}

void ProcUpdMulti::update_cpu(){
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
	
	for(auto it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_all_cpu();
		(*it)->update_ssi_cpu();
		(*it)->update_jxx_cpu();
		(*it)->update_row_cpu();
	}
	for(auto it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_rnd_cpu();
		(*it)->update_sup_cpu();
		(*it)->fill_spike();
	}
	for(auto it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_rcv_cpu();
		(*it)->update_que_cpu();
		(*it)->update_ssj_cpu();
		(*it)->update_col_cpu();
	}
	for(auto it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_rcv_cpu();
	}
}

#ifndef CPU_ONLY

void ProcUpdMulti::update_gpu(){
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

	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	
	for(auto it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_all_gpu();
		(*it)->update_ssi_gpu();
		(*it)->update_jxx_gpu();
		(*it)->update_row_gpu();
	}
	for(auto it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_rnd_gpu();
		(*it)->update_sup_gpu();
		(*it)->fill_spike();
	}

	for(int i=0; i<device_count; i++){
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
	for(auto it=_list_proj.begin(); it!=_list_proj.end(); it++){
		(*it)->update_rcv_gpu();
		(*it)->update_que_gpu();
		(*it)->update_ssj_gpu();
		(*it)->update_col_gpu();
	}
	for(auto it=_list_pop.begin(); it!=_list_pop.end(); it++){
		(*it)->update_rcv_gpu();
	}
	for(int i=0; i<device_count; i++){
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
}

#endif

}
}
