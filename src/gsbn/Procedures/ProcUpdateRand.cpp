#include "gsbn/procedures/ProcUpdRand.hpp"

namespace gsbn{

REGISTERIMPL(ProcUpdRand);

void ProcUpdRand::init_new(NetParam net_param, Database& db){
	init_copy(db);
}
void ProcUpdRand::init_copy(Database& db){
	CHECK(_hcu = db.table("hcu"));
	CHECK(_mcu = db.table("mcu"));
	int height = _mcu->height();
	
	CHECK(_rnd_uniform01 = db.create_table("rnd_uniform01", Table::SHARED, {sizeof(float)}, height));
	CHECK(_rnd_normal = db.create_table("rnd_normal", Table::SHARED, {sizeof(float)}, height));
	_rnd_uniform01->expand(height);
	_rnd_normal->expand(height);
}

void ProcUpdRand::update_cpu(){
	int size = _rnd_uniform01->height();
	_rnd.gen_uniform01_cpu(static_cast<float *>(_rnd_uniform01->mutable_cpu_data()), _rnd_uniform01->height());
	
	int h_hcu = _hcu->height();
	int idx=0;
	for(int i=0; i<h_hcu; i++){
		int mcu_num = static_cast<const int *>(_hcu->cpu_data(i))[Database::IDX_HCU_MCU_NUM];
		float snoise = static_cast<const float *>(_hcu->cpu_data(i))[Database::IDX_HCU_SNOISE];
		_rnd.gen_normal_cpu(static_cast<float *>(_rnd_normal->mutable_cpu_data(idx)), mcu_num, 0, snoise);
		idx += mcu_num;
	}
}

}
