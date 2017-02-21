#ifndef __GSBN_PROC_UPD_LAZY_2_POP_HPP__
#define __GSBN_PROC_UPD_LAZY_2_POP_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcUpdLazy2/Msg.hpp"

namespace gsbn{
namespace proc_upd_lazy_2{

class Pop{

public:
	Pop(){
		#ifndef CPU_ONLY
		cudaStreamCreate(&_stream);
		#endif
	};
	~Pop(){
		#ifndef CPU_ONLY
		cudaStreamDestroy(_stream);
		#endif
	};
	
	void init_new(int max_cycle, ProcParam proc_param, PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, Msg *msg);
	void init_copy(int max_cycle, ProcParam proc_param, PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, Msg *msg);
	void update_rnd_cpu();
	void update_sup_cpu();
	void fill_spike(int cycle);
	void send_spike(int cycle);
	#ifndef CPU_ONLY
	void update_rnd_gpu(int cycle);
	void update_sup_gpu(int cycle);
	#endif
	
	int _id;
	
	int _dim_proj;
	int _dim_hcu;
	int _dim_mcu;
	int _hcu_start;
	int _mcu_start;
	
	int _slot_num;
	int _fanout_num;
	
	Random _rnd;
	Msg* _msg;
	
	vector<int> _avail_proj;	// for src pop
	vector<vector<vector<int>>> _avail_hcu;	// for src pop
	
	SyncVector<int>* _slot;
	SyncVector<int>* _fanout;
	SyncVector<float>* _epsc;
	SyncVector<float>* _bj;
	SyncVector<float>* _dsup;
	SyncVector<float>* _act;
	SyncVector<int8_t>* _spike;
	SyncVector<float>* _rnd_uniform01;
	SyncVector<float>* _rnd_normal;
	SyncVector<float>* _wmask;
	SyncVector<float>* _lginp;
	Table* _conf;
	
	vector<Pop*>* _list_pop;
	
	float _taumdt;
	float _wtagain;
	float _maxfqdt;
	float _igain;
	float _wgain;
	float _lgbias;
	float _snoise;
	
	bool _flag_ext_spike;
	map<int, vector<int>> _ext_spikes;
	
	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
};

}

}

#endif //__GSBN_PROC_UPD_LAZY_2_POP_HPP__
