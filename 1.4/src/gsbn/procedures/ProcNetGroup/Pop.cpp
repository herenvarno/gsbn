#include "gsbn/procedures/ProcNetGroup/Pop.hpp"

namespace gsbn{
namespace proc_net_group{

void Pop::init_new(PopParam pop_param, Database& db, vector<Pop*>* list_pop, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){
	CHECK(list_pop);
	CHECK(list_group);
	CHECK(list_hcu);
	
	_list_pop=list_pop;
	_list_group=list_group;
	_list_hcu=list_hcu;
	
	_id=_list_pop->size();
	_hcu_start=_list_hcu->size();
	
	SyncVector<int>* spike;
	CHECK(spike = db.sync_vector_i("spike"));
	_mcu_start = spike->cpu_vector()->size();
	
	list_pop->push_back(this);
	
	_hcu_num=0;
	_mcu_num=0;
	int hcu_param_size = pop_param.hcu_param_size();
	for(int k=0; k<hcu_param_size; k++){
		HcuParam hcu_param = pop_param.hcu_param(k);
		int hcu_num = hcu_param.hcu_num();
		Group* g=new Group();
		g->init_new(hcu_param, db, _list_group, list_hcu, list_conn, msg);
		_hcu_num+=g->_hcu_num;
		_mcu_num+=g->_mcu_num;
/*
		for(int l=0;l<hcu_num;l++){
			Hcu* h = new Hcu();
			h->init_new(hcu_param, db, _list_hcu, list_conn, msg);
			_hcu_num+=1;
			_mcu_num+=h->_mcu_num;
		}*/
	}
}

void Pop::init_copy(PopParam pop_param, Database& db, vector<Pop*>* list_pop, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){
	CHECK(list_pop);
	CHECK(list_hcu);
	
	_list_pop=list_pop;
	_list_hcu=list_hcu;
	
	_id=_list_pop->size();
	_hcu_start=_list_hcu->size();
	
	SyncVector<int>* spike;
	CHECK(spike = db.sync_vector_i(".fake_spike")); // USE FAKE SPIKE VECTOR TO COUNT MCU NUM
	_mcu_start = spike->cpu_vector()->size();
	
	list_pop->push_back(this);
	
	_hcu_num=0;
	_mcu_num=0;
	int hcu_param_size = pop_param.hcu_param_size();
	for(int k=0; k<hcu_param_size; k++){
		HcuParam hcu_param = pop_param.hcu_param(k);
		int hcu_num = hcu_param.hcu_num();
		for(int l=0;l<hcu_num;l++){
			Hcu* h = new Hcu();
			h->init_copy(hcu_param, db, _list_hcu, list_conn, msg);
			_hcu_num+=1;
			_mcu_num+=h->_mcu_num;
		}
	}
	CHECK(spike = db.sync_vector_i("spike")); // CHANGE TO REAL SPIKE VECTOR
}

Hcu* Pop::get_hcu(int i){
	CHECK_LT(i, _hcu_num);
	return (*_list_hcu)[_hcu_start+i];
}

}
}
