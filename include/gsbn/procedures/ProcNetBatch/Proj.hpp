#ifndef __GSBN_PROC_NET_BATCH_PROJ_HPP__
#define __GSBN_PROC_NET_BATCH_PROJ_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcNetBatch/Pop.hpp"
#include "gsbn/procedures/ProcNetBatch/Msg.hpp"


namespace gsbn{
namespace proc_net_batch{

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
	
	void init_new(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg);
	void init_copy(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg);
	
	void update_full_cpu();
	void update_j_cpu();
	void update_ss_cpu();
	void update_row_cpu();
	void update_col_cpu();
	void receive();
	void add_row(int src_mcu, int dest_hcu, int delay);
	#ifndef CPU_ONLY
	void update_full_gpu();
	void update_j_gpu();
	void update_ss_gpu();
	void update_row_gpu();
	void update_col_gpu();
	#endif

	int _id;
	
	int _proj_in_pop;
	int _dim_conn;
	int _dim_hcu;
	int _dim_mcu;
	
	Random _rnd;
	Msg* _msg;
	
	vector<Proj*>* _list_proj;
	vector<Pop*>* _list_pop;
	vector<vector<int>> _avail_hcu;
	vector<int> _conn_cnt;
	
	Pop* _ptr_src_pop;
	Pop* _ptr_dest_pop;
	
	SyncVector<int>* _ii;
	SyncVector<int>* _di;
	SyncVector<int>* _qi;
	SyncVector<float>* _pi;
	SyncVector<float>* _ei;
	SyncVector<float>* _zi;
	SyncVector<int>* _ti;
	SyncVector<float>* _pj;
	SyncVector<float>* _ej;
	SyncVector<float>* _zj;
	SyncVector<float>* _pij;
	SyncVector<float>* _eij;
	SyncVector<float>* _zi2;
	SyncVector<float>* _zj2;
	SyncVector<int>* _tij;
	SyncVector<float>* _wij;
	SyncVector<int>* _ssi;
	SyncVector<int>* _ssj;

	SyncVector<float>* _epsc;
	SyncVector<float>* _bj;
	
	SyncVector<int>* _si;
	SyncVector<int>* _sj;
	Table* _conf;

	float _taupdt;
	float _tauedt;
	float _tauzidt;
	float _tauzjdt;
	float _eps;
	float _eps2;
	float _kfti;
	float _kftj;
	float _wgain;
	float _bgain;
	float _pi0;
	
	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
};

}
}

#endif //__GSBN_PROC_NET_CONN_HPP__