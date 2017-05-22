#ifndef __GSBN_PROC_UPD_MULTI_POP_HPP__
#define __GSBN_PROC_UPD_MULTI_POP_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Parser.hpp"
#include "gsbn/GlobalVar.hpp"

namespace gsbn{
namespace proc_upd_multi{

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
	
	void init_new(ProcParam proc_param, PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt);
	void init_copy(ProcParam proc_param, PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt);
	void update_rnd_cpu();
	void update_sup_cpu();
	void fill_spike();
	#ifndef CPU_ONLY
	void update_rnd_gpu();
	void update_sup_gpu();
	#endif
	
	int _id;
	int _rank;
	int _device;
	
	int _dim_proj;
	int _dim_hcu;
	int _dim_mcu;
	int _hcu_start;
	int _mcu_start;
	
	Random _rnd;
	
	SyncVector<float>* _epsc;
	SyncVector<float>* _bj;
	SyncVector<float>* _dsup;
	SyncVector<float>* _act;
	SyncVector<float>* _ada;
	SyncVector<int8_t>* _spike;
	SyncVector<float>* _rnd_uniform01;
	SyncVector<float>* _rnd_normal;
	SyncVector<float>* _wmask;
	SyncVector<float>* _lginp;
	SyncVector<int>* _counter;
	GlobalVar _glv;
	
	vector<Pop*>* _list_pop;
	
	float _taumdt;
	float _wtagain;
	float _maxfqdt;
	float _igain;
	float _wgain;
	float _lgbias;
	float _snoise;
	float _adgain;
	float _tauadt;
	
	bool _flag_ext_spike;
	map<int, vector<int>> _ext_spikes;
	
	int _spike_buffer_size;
	
	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
};

}

}

#endif //__GSBN_PROC_UPD_MULTI_POP_HPP__
