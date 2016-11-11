#include "gsbn/procedures/ProcRnd.hpp"

namespace gsbn{

REGISTERIMPL(ProcRnd);

void ProcRnd::init_new(NetParam net_param, Database& db){
	int hcu_count=0;
	int mcu_count=0;

	// pop, hcu, mcu, hcu_slot, mcu_fanout, spk, sup, addr
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			int hcu_param_size = pop_param.hcu_param_size();
			int total_hcu_num=0;
			for(int k=0; k<hcu_param_size; k++){
				HcuParam hcu_param = pop_param.hcu_param(k);
				int hcu_num = hcu_param.hcu_num();
				total_hcu_num+=hcu_num;
				int mcu_in_hcu=0;
				_mcu_start.push_back(mcu_count);
				
				for(int l=0;l<hcu_num;l++){
					int mcu_param_size = hcu_param.mcu_param_size();
					int total_mcu_num=0;
					hcu_count++;
					
					for(int m=0; m<mcu_param_size; m++){
						McuParam mcu_param = hcu_param.mcu_param(m);
						int mcu_num = mcu_param.mcu_num();
						total_mcu_num += mcu_num;
						mcu_count += mcu_num;
					}
					mcu_in_hcu += total_mcu_num;
				}
				
				_snoise.push_back(hcu_param.snoise());
				_mcu_num.push_back(mcu_in_hcu);
			}
		}
	}
	
	CHECK(_rnd_uniform01 = db.create_sync_vector_f(".rnd_uniform01"));
	CHECK(_rnd_normal = db.create_sync_vector_f(".rnd_normal"));
	_rnd_uniform01->mutable_cpu_vector()->resize(mcu_count);
	_rnd_normal->mutable_cpu_vector()->resize(mcu_count);
}

void ProcRnd::init_copy(NetParam net_param, Database& db){
	init_new(net_param, db);
}

void ProcRnd::update_cpu(){
	HOST_VECTOR(float, *v_uniform01) = _rnd_uniform01->mutable_cpu_vector();
	HOST_VECTOR(float, *normal) = _rnd_normal->mutable_cpu_vector();
	
	float *ptr_uniform01= _rnd_uniform01->mutable_cpu_data();
	float *ptr_normal= _rnd_normal->mutable_cpu_data();
	int size_uniform01 = v_uniform01->size();
	_rnd.gen_uniform01_cpu(ptr_uniform01, size_uniform01);
        
	int l = _mcu_start.size();
	for(int i=0; i<l; i++){
		_rnd.gen_normal_cpu(ptr_normal+_mcu_start[i], _mcu_num[i], 0, _snoise[i]);
	}
}

#ifndef CPU_ONLY
void ProcRnd::update_gpu(){
	DEVICE_VECTOR(float, *v_uniform01) = _rnd_uniform01->mutable_gpu_vector();
	
	float *ptr_uniform01= _rnd_uniform01->mutable_gpu_data();
	float *ptr_normal= _rnd_normal->mutable_gpu_data();
	int size_uniform01 = v_uniform01->size();
	_rnd.gen_uniform01_gpu(ptr_uniform01, size_uniform01);
        
	int l = _mcu_start.size();
	for(int i=0; i<l; i++){
		_rnd.gen_normal_gpu(ptr_normal+_mcu_start[i], _mcu_num[i], 0, _snoise[i]);
	}
}
#endif

}
