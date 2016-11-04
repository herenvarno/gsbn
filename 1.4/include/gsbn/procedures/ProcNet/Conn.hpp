#ifndef __GSBN_PROC_NET_CONN_HPP__
#define __GSBN_PROC_NET_CONN_HPP__

#include "gsbn/Database.hpp"

namespace gsbn{
namespace proc_net{

class Conn{

public:
	Conn(){
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaStreamCreate(&_stream));
		#endif
	};
	~Conn(){
		#ifndef CPU_ONLY
		CUDA_CHECK(cudaStreamDestroy(_stream));
		#endif
	};
	
	void init_new(ProjParam proj_param, Database& db, vector<Conn*>* list_conn, int w);
	void init_copy(ProjParam proj_param, Database& db, vector<Conn*>* list_conn, int w);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu_1();
	void update_gpu_2();
	void update_gpu_3();
	void update_gpu_4();
	void update_gpu_5();
	void update_gpu_6();
	void update_gpu_7();
	void update_gpu();
	#endif
	
	void add_row_cpu(int src_mcu, int delay);
	#ifndef CPU_ONLY
	void add_row_gpu(int src_mcu, int delay);
	#endif
public:
	
	int _id;
	int _proj_start;
	int _mcu_start;
	int _h;
	int _w;
	
	SyncVector<int>* _ii;
	SyncVector<int>* _qi;
	SyncVector<int>* _di;
	SyncVector<int>* _si;
	SyncVector<int>* _ssi;
	SyncVector<float>* _pi;
	SyncVector<float>* _ei;
	SyncVector<float>* _zi;
	SyncVector<int>* _ti;
	SyncVector<int>* _sj;
	SyncVector<int>* _ssj;
	SyncVector<float>* _pj;
	SyncVector<float>* _ej;
	SyncVector<float>* _zj;
//	SyncVector<int>* _tj;
	SyncVector<float>* _pij;
	SyncVector<float>* _eij;
	SyncVector<float>* _zi2;
	SyncVector<float>* _zj2;
	SyncVector<int>* _tij;
	SyncVector<float>* _wij;
	
	#ifndef CPU_ONLY
	cudaStream_t _stream;
	#endif
	
	// External
	SyncVector<float> *_epsc;
	SyncVector<float> *_bj;
	SyncVector<int> *_spike;
	
	Table *_conf;
	
	float _tauzidt;
	float _tauzjdt;
	float _tauedt;
	float _taupdt;
	float _eps;
	float _eps2;
	float _kfti;
	float _kftj;
	float _bgain;
	float _wgain;
	float _pi0;
	
	vector<Conn*>* _list_conn;

};

}

}

#endif //__GSBN_PROC_NET_CONN_HPP__
