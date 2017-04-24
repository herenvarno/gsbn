#include "gsbn/procedures/ProcSpkRec/Pop.hpp"

namespace gsbn{
namespace proc_spk_rec{

Pop::Pop(int& id, PopParam pop_param, Database& db){
	_id = id;
	_rank = pop_param.rank();
	_dim_hcu = pop_param.hcu_num();
	_dim_mcu = pop_param.mcu_num();
	_maxfq = pop_param.maxfq();
	
	// DO NOT CHECK THE RETURN VALUE, SINCE THE SPIKE VECTOR MAYBE NOT IN THE CURRENT
	// RANK.
	_spike = db.sync_vector_i8("spike_" + to_string(_id));
	
	_spike_buffer_size=1;
	if(_spike){
		if(_spike->ld()>0){
			_spike_buffer_size = _spike->size()/_spike->ld();
		}
	}
	
	id++;
}

Pop::~Pop(){
}

void Pop::record(string filename, int simstep){
	int cursor = 0;
	if(_spike_buffer_size > 1){
		cursor = simstep % _spike_buffer_size;
	}
	
	const int8_t *ptr_spike = _spike->cpu_data()+cursor*_dim_hcu*_dim_mcu;
	fstream output(filename, ios::out | ios::app);
	
	bool flag=false;
	for(int i=0; i<_dim_hcu*_dim_mcu; i++){
		if(ptr_spike[i]>0){
			if(!flag){
				output << simstep;
				flag = true;
			}
			output << "," << i;
		}
	}
	if(flag){
		output<<endl;
	}
}

}
}
