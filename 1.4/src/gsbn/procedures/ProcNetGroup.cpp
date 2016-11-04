#include "gsbn/procedures/ProcNetGroup.hpp"

namespace gsbn{
namespace proc_net_group{

REGISTERIMPL(ProcNetGroup);

void ProcNetGroup::init_new(NetParam net_param, Database& db){
	CHECK(_spike = db.create_sync_vector_i("spike"));
	
	_msg.init_new(net_param, db);
	LOG(INFO) <<"HERE1";
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_new(pop_param, db, &_list_pop, &_list_group, &_list_hcu, &_list_conn, &_msg);
		}
	}
	
	LOG(INFO) <<"HERE2";
	int total_pop_num = _list_pop.size();
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			Proj *proj = new Proj();
			proj->init_new(proj_param, db, &_list_proj, &_list_pop, &_list_group,  &_list_hcu, &_list_conn);
		}
	}
	LOG(INFO) <<"HERE3";
	_list_thread_hcu.resize(_list_hcu.size());
	_list_thread_conn.resize(_list_conn.size());
}

void ProcNetGroup::init_copy(NetParam net_param, Database& db){
	SyncVector<int> *fake_spike;
	CHECK(fake_spike = db.create_sync_vector_i(".fake_spike")); // USED TO COUNT MCU NUMBER
	CHECK(_spike = db.sync_vector_i("spike"));
	
	_msg.init_copy(net_param, db);

	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop *p = new Pop();
			p->init_copy(pop_param, db, &_list_pop, &_list_group, &_list_hcu, &_list_conn, &_msg);
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
			proj->init_copy(proj_param, db, &_list_proj, &_list_pop, &_list_group, &_list_hcu, &_list_conn);
		}
	}
	
	_list_thread_hcu.resize(_list_hcu.size());
	_list_thread_conn.resize(_list_conn.size());
	CHECK_EQ(_spike->cpu_vector()->size(), fake_spike->cpu_vector()->size());
}

static void* func_thread_hcu_cpu(void* ptr){
	Hcu* p=(Hcu*)(ptr);
	p->update_cpu();
}
static void* func_thread_conn_cpu(void* ptr){
	Conn* p=(Conn*)(ptr);
	p->update_cpu();
}
void ProcNetGroup::update_cpu(){
	_msg.update();
	
	for(vector<Group*>::iterator it=_list_group.begin(); it!=_list_group.end(); it++){
		(*it)->update_cpu();
	}
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		LOG(INFO) << "here!!";
		(*it)->update_cpu();
	}
	for(vector<Hcu*>::iterator it=_list_hcu.begin(); it!=_list_hcu.end(); it++){
		LOG(INFO) << "here2!!";
		(*it)->send_receive_cpu();
	}
}

#ifndef CPU_ONLY
static void* func_thread_hcu_gpu(void* ptr){
	Hcu* p=(Hcu*)(ptr);
	p->update_gpu();
}
static void* func_thread_conn_gpu(void* ptr){
	Conn* p=(Conn*)(ptr);
	p->update_gpu();
}

void ProcNetGroup::update_gpu(){
	_msg.update();

	for(vector<Group*>::iterator it=_list_group.begin(); it!=_list_group.end(); it++){
		(*it)->update_gpu();
	}
	cudaDeviceSynchronize();
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		(*it)->update_gpu_1();
	}
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		(*it)->update_gpu_2();
	}
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		(*it)->update_gpu_3();
	}
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		(*it)->update_gpu_4();
	}
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		(*it)->update_gpu_5();
	}
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		(*it)->update_gpu_6();
	}
	for(vector<Conn*>::iterator it=_list_conn.begin(); it!=_list_conn.end(); it++){
		(*it)->update_gpu_7();
	}
	for(vector<Hcu*>::iterator it=_list_hcu.begin(); it!=_list_hcu.end(); it++){
		(*it)->send_receive_gpu();
	}
}
#endif

}
}
