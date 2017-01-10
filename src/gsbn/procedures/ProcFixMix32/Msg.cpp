#include "gsbn/procedures/ProcFixMix32/Msg.hpp"

namespace gsbn{
namespace proc_fix_mix32{

enum{
	IDX_PROJ,
	IDX_SRC_MCU,
	IDX_DEST_HCU,
	IDX_TYPE,
	IDX_DELAY,
	IDX_QUEUE,
	IDX_COUNT
};

void Msg::init_new(NetParam net_param, Database& db){
	CHECK(_msgbox = db.create_sync_vector_i32("msg"));
	int proj_param_size = net_param.proj_param_size();
	_list_active_msg.resize(proj_param_size);
}
void Msg::init_copy(NetParam net_param, Database& db){
	CHECK(_msgbox = db.sync_vector_i32("msg"));
	int proj_param_size = net_param.proj_param_size();
	_list_active_msg.resize(proj_param_size);
	
	for(HOST_VECTOR_ITERATOR(int, it) = _msgbox->mutable_cpu_vector()->begin(); it<_msgbox->mutable_cpu_vector()->end(); it+=7){
		if((*(it+IDX_TYPE)==0) || (*(it+IDX_QUEUE)==0)){
			#ifndef CPU_ONLY
			_empty_pos.push_back(thrust::distance(_msgbox->mutable_cpu_vector()->begin(), it));
			#else
			_empty_pos.push_back(std::distance(_msgbox->mutable_cpu_vector()->begin(), it));
			#endif
		}
	}
}


void Msg::send(int proj, int src_mcu, int dest_hcu, int type){
	HOST_VECTOR(int, *v_msgbox) = _msgbox->mutable_cpu_vector();
	if(_empty_pos.empty()){
		v_msgbox->push_back(proj);
		v_msgbox->push_back(src_mcu);
		v_msgbox->push_back(dest_hcu);
		v_msgbox->push_back(type);
		v_msgbox->push_back(calc_delay(src_mcu, dest_hcu));	// delay
		v_msgbox->push_back(0x01 << calc_delay(src_mcu, dest_hcu));
	}else{
		int offset = _empty_pos[_empty_pos.size()-1];
		_empty_pos.pop_back();
		(*v_msgbox)[offset+IDX_PROJ] = proj;
		(*v_msgbox)[offset+IDX_SRC_MCU] = src_mcu;
		(*v_msgbox)[offset+IDX_DEST_HCU] = dest_hcu;
		(*v_msgbox)[offset+IDX_TYPE] = type;
		(*v_msgbox)[offset+IDX_DELAY] = calc_delay(src_mcu, dest_hcu);
		(*v_msgbox)[offset+IDX_QUEUE] = 0x01 << calc_delay(src_mcu, dest_hcu);
	}
}

vector<msg_t> Msg::receive(int hcu_id){
	return _list_active_msg[hcu_id];
}

void Msg::clear_empty_pos(){
	HOST_VECTOR(int, *v_msgbox) = _msgbox->mutable_cpu_vector();
	HOST_VECTOR_ITERATOR(int, it_msgbox) = _msgbox->mutable_cpu_vector()->begin();
	int l=_empty_pos.size();
	for(int i=0; i<l; i++){
		v_msgbox->erase(it_msgbox+_empty_pos[i], it_msgbox+_empty_pos[i]+IDX_COUNT);
	}
	_empty_pos.clear();
}

int Msg::calc_delay(int src_mcu, int dest_hcu){
	return 1;
}

void Msg::update(){
	int list_size=_list_active_msg.size();
	for(int i=0; i<list_size; i++){
		_list_active_msg[i].clear();
	}
	
	int i=0;
	for(HOST_VECTOR_ITERATOR(int, it) = _msgbox->mutable_cpu_vector()->begin(); it<_msgbox->mutable_cpu_vector()->end(); it+=IDX_COUNT){
		i++;
		msg_t m;
		m.type = it[IDX_TYPE];
	
		it[IDX_QUEUE] >>= 1;

		if((it[IDX_QUEUE]&0x01) && m.type){
			m.proj = it[IDX_PROJ];
			m.src_mcu = it[IDX_SRC_MCU];
			m.dest_hcu = it[IDX_DEST_HCU];
			m.delay = it[IDX_DELAY];
			_list_active_msg[m.proj].push_back(m);
			it[IDX_TYPE]=0;
			#ifndef CPU_ONLY
			_empty_pos.push_back(thrust::distance(_msgbox->mutable_cpu_vector()->begin(), it));
			#else
			_empty_pos.push_back(std::distance(_msgbox->mutable_cpu_vector()->begin(), it));
			#endif
		}
	}
}

}
}
