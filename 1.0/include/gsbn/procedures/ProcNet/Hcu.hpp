#ifndef __GSBN_PROC_NET_HCU_HPP__
#define __GSBN_PROC_NET_HCU_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcNet/Conn.hpp"
#include "gsbn/procedures/ProcNet/Msg.hpp"

namespace gsbn{
namespace proc_net{

class Hcu{

public:
	Hcu(HcuParam hcu_param, Database& db, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg);
	~Hcu();
	
	void update_cpu();
	void send_receive();

public:
	int _id;
	
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
};

}

}

#endif //__GSBN_PROC_NET_HCU_HPP__
