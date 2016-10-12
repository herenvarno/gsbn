#include "gsbn/procedures/ProcNet/Pop.hpp"

namespace gsbn{
namespace proc_net{

void Pop::init_new(PopParam pop_param, Database& db, vector<Pop*>* list_pop, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){
	CHECK(list_pop);
	CHECK(list_hcu);
	
	_list_pop=list_pop;
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
		for(int l=0;l<hcu_num;l++){
			Hcu* h = new Hcu(hcu_param, db, _list_hcu, list_conn, msg);
			_hcu_num+=1;
			_mcu_num+=h->_mcu_num;
		}
	}
	
	LOG(INFO) << "++++++++++++++ pop id=" << _id << " ++++++++++++++" << endl;
	
	LOG(INFO) << "hcu_start=" << _hcu_start;
	LOG(INFO) << "hcu_num=" << _hcu_num;
	LOG(INFO) << "mcu_start=" << _mcu_start;
	LOG(INFO) << "mcu_num=" << _mcu_num;
}

void Pop::init_copy(PopParam pop_param, Database& db, vector<Pop*>* list_pop, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg){
	init_new(pop_param, db, list_pop, list_hcu, list_conn, msg);
}

Hcu* Pop::get_hcu(int i){
	CHECK_LT(i, _hcu_num);
	return (*_list_hcu)[_hcu_start+i];
}

}
}
