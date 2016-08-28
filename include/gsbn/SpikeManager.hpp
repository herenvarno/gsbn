#ifndef __GSBN_SPIKE_MANAGER_HPP__
#define __GSBN_SPIKE_MANAGER_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"

namespace gsbn{

class SpikeManager{

public:
	SpikeManager();
	void init(Database& db);
	void learn(int timestamp, int stim_offset);
	void recall(int timestamp);
	
	
private:
	void update_phase_1();
	void update_phase_2(int stim_offset);
	void update_phase_3();
	void update_phase_4();
	void update_phase_5();
	void update_phase_6();
	
	Table* _j_array;
	Table* _spk;
	Table* _hcu;
	Table* _sup;
	Table* _stim;
	Table* _mcu;
	Table* _tmp1;
	Table* _epsc;
	Table* _conf;
	Table* _addr;
	
	float _kftj, _kp, _ke, _kzi, _kzj, _bgain, _eps, _maxfqdt, _wtagain, _taumdt, _snoise, _igain, _lgbias, _wgain;
};

}

#endif //__GSBN_SPIKE_MANAGER_HPP__
