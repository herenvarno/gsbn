#ifndef __GSBN_PROC_HALF_POP_HPP__
#define __GSBN_PROC_HALF_POP_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcHalf/Msg.hpp"

namespace gsbn{
namespace proc_half{

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
	
	void init_new(PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, Msg *msg);
	void init_copy(PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, Msg *msg);
	void update_rnd_cpu();
	void update_sup_cpu();
	void send();
	#ifndef CPU_ONLY
	void update_rnd_gpu();
	void update_sup_gpu();
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
	SyncVector<fp16>* _epsc;
	SyncVector<fp16>* _bj;
	SyncVector<fp16>* _dsup;
	SyncVector<fp16>* _act;
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
	
	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
};

}

}

#endif //__GSBN_PROC_HALF_POP_HPP__
