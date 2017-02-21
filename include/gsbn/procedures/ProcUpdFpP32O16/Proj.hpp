#ifndef __GSBN_PROC_UPD_FP_P32_O16_PROJ_HPP__
#define __GSBN_PROC_UPD_FP_P32_O16_PROJ_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcUpdFpP32O16/Pop.hpp"
#include "gsbn/procedures/ProcUpdFpP32O16/Msg.hpp"


namespace gsbn{
namespace proc_upd_fp_p32_o16{

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
	
	void init_new(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg);
	void init_copy(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg);
	
	void update_full_cpu();
	void update_j_cpu();
	void update_ss_cpu();
	void update_row_cpu();
	void update_col_cpu();
	void receive_spike();
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
	SyncVector<fp16>* _ei;
	SyncVector<fp16>* _zi;
	SyncVector<int>* _ti;
	SyncVector<float>* _pj;
	SyncVector<fp16>* _ej;
	SyncVector<fp16>* _zj;
	SyncVector<float>* _pij;
	SyncVector<fp16>* _eij;
	SyncVector<fp16>* _zi2;
	SyncVector<fp16>* _zj2;
	SyncVector<int>* _tij;
	SyncVector<fp16>* _wij;
	SyncVector<int>* _ssi;
	SyncVector<int>* _ssj;

	SyncVector<fp16>* _epsc;
	SyncVector<fp16>* _bj;
	
	SyncVector<int8_t>* _si;
	SyncVector<int8_t>* _sj;
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

#endif //__GSBN_PROC_UPD_FP_P32_O16_PROJ_HPP__
