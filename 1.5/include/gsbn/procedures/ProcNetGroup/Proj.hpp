#ifndef __GSBN_PROC_NET_GROUP_PROJ_HPP__
#define __GSBN_PROC_NET_GROUP_PROJ_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcNetGroup/Conn.hpp"
#include "gsbn/procedures/ProcNetGroup/Hcu.hpp"
#include "gsbn/procedures/ProcNetGroup/Pop.hpp"
#include "gsbn/procedures/ProcNetGroup/Group.hpp"


namespace gsbn{
namespace proc_net_group{

class Proj{

public:
	Proj(){
		#ifndef CPU_ONLY
		cudaStreamCreate(&_stream);
		#endif
	};
	~Proj(){
		#ifndef CPU_ONLY
		cudaStreamDestroy(_stream);
		#endif
	};
	
	void init_new(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop,  vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn);
	void init_copy(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn);
	
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

	int _offset_in_pop;
	int _offset_in_spk;
	int _mcu_num;
	
	vector<Proj*>* _list_proj;
	Pop* _ptr_src_pop;
	Pop* _ptr_dest_pop;

	SyncVector<float>* _pj;
	SyncVector<float>* _ej;
	SyncVector<float>* _zj;
	SyncVector<int>* _sj;
	SyncVector<float>* _epsc;
	SyncVector<float>* _bj;
	Table* _conf;

	float _taupdt;
	float _tauedt;
	float _tauzidt;
	float _tauzjdt;
	float _eps;
	float _kftj;
	float _bgain;
	
	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
};

}

}

#endif //__GSBN_PROC_NET_CONN_HPP__
