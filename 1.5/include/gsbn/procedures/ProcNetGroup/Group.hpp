#ifndef __GSBN_PROC_NET_GROUP_GROUP_HPP__
#define __GSBN_PROC_NET_GROUP_GROUP_HPP__

#include "gsbn/procedures/ProcNetGroup/Hcu.hpp"

namespace gsbn{
namespace proc_net_group{

class Group{
public:
	Group(){
		#ifndef CPU_ONLY
		cudaStreamCreate(&_stream);
		#endif
	};
	~Group(){
		#ifndef CPU_ONLY
		cudaStreamDestroy(_stream);
		#endif
	};
	
	void init_new(HcuParam hcu_param, Database& db, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg);
	void init_copy(HcuParam hcu_param, Database& db, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg);
	
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

public:
	int _id;
	int _hcu_start;
	int _hcu_num;
	int _mcu_start;
	int _mcu_num;
	int _conn_num;
	int _mcu_start_in_pop;
	int _mcu_num_in_pop;
	
	SyncVector<float>* _epsc;
	SyncVector<float>* _bj;
	SyncVector<float>* _dsup;
	SyncVector<float>* _act;

	float _taumdt;
	float _wtagain;
	float _maxfqdt;
	float _igain;
	float _wgain;
	float _lgbias;

	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
	
	SyncVector<int>* _spike;
	SyncVector<float>* _rnd_uniform01;
	SyncVector<float>* _rnd_normal;
	SyncVector<float>* _lginp;
	SyncVector<float>* _wmask;

	Table* _conf;

	vector<Hcu*> _list_hcu;

};

}
}

#endif //__GSBN_PROC_NET_GROUP_HPP__
