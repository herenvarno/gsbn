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
	_proj_num = 0;
	_hcu_start=_list_hcu->size();
	
	SyncVector<int>* spike;
	CHECK(spike = db.sync_vector_i("spike"));
	_mcu_start = spike->cpu_vector()->size();
	
	list_pop->push_back(this);
	
	vector<Group*> lst_grp;
	_hcu_num=0;
	_mcu_num=0;
	int hcu_param_size = pop_param.hcu_param_size();
	for(int k=0; k<hcu_param_size; k++){
		HcuParam hcu_param = pop_param.hcu_param(k);
		int hcu_num = hcu_param.hcu_num();
		Group* g=new Group();
		g->init_new(hcu_param, db, _list_group, list_hcu, list_conn, msg);
		lst_grp.push_back(g);
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
	
	CHECK(_pj=db.create_sync_vector_f("pj_"+to_string(_id)));
	CHECK(_ej=db.create_sync_vector_f("ej_"+to_string(_id)));
	CHECK(_zj=db.create_sync_vector_f("zj_"+to_string(_id)));
	CHECK(_epsc=db.create_sync_vector_f("epsc_"+to_string(_id)));
	CHECK(_bj=db.create_sync_vector_f("bj_"+to_string(_id)));
	for(vector<Group*>::iterator it=lst_grp.begin(); it!=lst_grp.end(); it++){
		(*it)->_mcu_num_in_pop = _mcu_num;
		(*it)->_mcu_start_in_pop = _mcu_start;
		(*it)->_epsc = _epsc;
		(*it)->_bj = _bj;
	}
	
}

// FIXME : init_copy need to update
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
