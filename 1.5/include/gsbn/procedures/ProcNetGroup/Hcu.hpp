#ifndef __GSBN_PROC_NET_GROUP_HCU_HPP__
#define __GSBN_PROC_NET_GROUP_HCU_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcNetGroup/Conn.hpp"
#include "gsbn/procedures/ProcNetGroup/Msg.hpp"

namespace gsbn{
namespace proc_net_group{

class Hcu{

public:
	Hcu(){
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaStreamCreate(&_stream));
		#endif
	};
	~Hcu(){
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaStreamDestroy(_stream));
		#endif
	};
	
	void init_new(HcuParam hcu_param, Database& db, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg);
	void init_copy(HcuParam hcu_param, Database& db, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu_1();
	void update_gpu_2();
	void update_gpu_2_1();
	void update_gpu_2_2();
	void update_gpu_2_3();
	void update_gpu_2_4();
	void update_gpu_3();
	void update_gpu();
	#endif

	void send_receive_cpu();
	#ifndef CPU_ONLY
	void send_receive_gpu();
	#endif

public:
	int _id;
	int _group_id;
	
	Random _rnd;
	
	vector<Conn*> _isp;
	vector<int> _isp_mcu_start;
	vector<int> _isp_mcu_num;
	vector<vector<int>> _avail_hcu;
	SyncVector<float>* _epsc;
	SyncVector<float>* _bj;
	SyncVector<float>* _dsup;
	SyncVector<float>* _act;

	SyncVector<int>* _slot;
	SyncVector<int>* _fanout;
	
	int _mcu_start;
	int _mcu_num;
	
	float _taumdt;
	float _wtagain;
	float _maxfqdt;
	float _igain;
	float _wgain;
	float _lgbias;
	
	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
	
	// set externally
	SyncVector<int>* _spike;
	SyncVector<float>* _rnd_uniform01;
	SyncVector<float>* _rnd_normal;
	SyncVector<float>* _lginp;
	SyncVector<float>* _wmask;
	Table* _conf;
	
	vector<Hcu*>* _list_hcu;
	vector<Conn*>* _list_conn;
	Msg* _msg;

	// temp
	float _maxdsup;
	float _vsum;
};

}

}

#endif //__GSBN_PROC_NET_HCU_HPP__
