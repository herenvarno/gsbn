#include "gsbn/procedures/ProcNet/Msg.hpp"

namespace gsbn{
namespace proc_net{

void Msg::init_new(NetParam net_param, Database& db){
	db.register_sync_vector_i("msg", &_msgbox);
	
	int hcu_count=0;

	// pop, hcu, mcu, hcu_slot, mcu_fanout, spk, sup, addr
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			int hcu_param_size = pop_param.hcu_param_size();
			for(int k=0; k<hcu_param_size; k++){
				HcuParam hcu_param = pop_param.hcu_param(k);
				int hcu_num = hcu_param.hcu_num();
				hcu_count+=hcu_num;
			}
		}
	}
	
	_list_active_msg.resize(hcu_count);
}
void Msg::init_copy(NetParam net_param, Database& db){
	__NOT_IMPLEMENTED__;
}


void Msg::send(int src_hcu, int src_mcu, int dest_hcu, int dest_mcu, int type){
	HOST_VECTOR(int, *v_msgbox) = _msgbox.mutable_cpu_vector();
	if(_empty_pos.empty()){
		v_msgbox->push_back(src_hcu);
		v_msgbox->push_back(src_mcu);
		v_msgbox->push_back(dest_hcu);
		v_msgbox->push_back(dest_mcu);
		v_msgbox->push_back(type);
		v_msgbox->push_back(calc_delay(src_hcu, dest_hcu));	// delay
		v_msgbox->push_back(0x01 << calc_delay(src_hcu, dest_hcu));
	}else{
		int offset = _empty_pos[_empty_pos.size()-1];
		_empty_pos.pop_back();
		(*v_msgbox)[offset+0] = src_hcu;
		(*v_msgbox)[offset+1] = src_mcu;
		(*v_msgbox)[offset+2] = dest_hcu;
		(*v_msgbox)[offset+3] = dest_mcu;
		(*v_msgbox)[offset+4] = type;
		(*v_msgbox)[offset+5] = calc_delay(src_hcu, dest_hcu);
		(*v_msgbox)[offset+6] = 0x01 << calc_delay(src_hcu, dest_hcu);
	}
}

vector<msg_t> Msg::receive(int hcu_id){
	return _list_active_msg[hcu_id];
}

void Msg::clear_empty_pos(){
	HOST_VECTOR(int, *v_msgbox) = _msgbox.mutable_cpu_vector();
	HOST_VECTOR_ITERATOR(int, it_msgbox) = _msgbox.mutable_cpu_vector()->begin();
	int l=_empty_pos.size();
	for(int i=0; i<l; i++){
		v_msgbox->erase(it_msgbox+_empty_pos[i], it_msgbox+_empty_pos[i]+7);
	}
	_empty_pos.clear();
}

int Msg::calc_delay(int src_hcu, int dest_hcu){
	return 1;
}

void Msg::update(){
	int list_size=_list_active_msg.size();
	for(int i=0; i<list_size; i++){
		_list_active_msg[i].clear();
	}
	
	int i=0;
	for(HOST_VECTOR_ITERATOR(int, it) = _msgbox.mutable_cpu_vector()->begin(); it<_msgbox.mutable_cpu_vector()->end(); it+=7){
		i++;
		msg_t m;
		m.type = *(it+4);
	
		(*(it+6)) >>= 1;
		
//		LOG(INFO)<<"["<< i <<"] #= "<<*(it+0) << "|" << *(it+1) << "|" <<*(it+2)<<"|" << *(it+3) << "|" <<*(it+4)<<"|" << *(it+5) << "|" <<*(it+6);
		if((*(it+6)&0x01) && m.type){
			m.src_hcu = *(it+0);
			m.src_mcu = *(it+1);
			m.dest_hcu = *(it+2);
			m.dest_mcu = *(it+3);
			m.delay = *(it+5);
			_list_active_msg[m.dest_hcu].push_back(m);
			*(it+4)=0;
			_empty_pos.push_back(distance(_msgbox.mutable_cpu_vector()->begin(), it));
		}
	}

//	if(_empty_pos.size()>10){
//		clear_empty_pos();
//	}
}

}
}
