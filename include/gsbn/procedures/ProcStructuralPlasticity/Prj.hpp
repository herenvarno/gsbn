#ifndef __GSBN_PROC_STRUCTURAL_PLASTICITY_PRJ_HPP__
#define __GSBN_PROC_STRUCTURAL_PLASTICITY_PRJ_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Parser.hpp"
#include "gsbn/GlobalVar.hpp"

#include "gsbn/procedures/ProcStructuralPlasticity/Pop.hpp"

namespace gsbn{
namespace proc_structural_plasticity{

class Prj{

public:
	Prj(int& id, vector<int>& shared_buffer_size_list, vector<Pop>& pop_list, ProjParam prj_param, Database& db, int rank);
	~Prj();
	
	void remove_conn(int row);
	void assign_conn(int *ptr_new_ii, int *ptr_new_di);
	vector<int> get_avail_active_hcu_list(int threshold);
	vector<int> prune(int threshold_t, float threshold_wp, float threshold_wn);
	
	int _rank;
	int _id;
	int _src_pop;
	int _dest_pop;
	int _dim_conn;
	int _dim_hcu;
	int _dim_mcu;
	int _hcu_start;
	int _mcu_start;
	int _shared_buffer_offset;
	
	SyncVector<int>* _local_buffer;
	
	SyncVector<int>* _ii;
	SyncVector<int>* _di;
	SyncVector<int>* _ti;
	SyncVector<float>* _pi;
	SyncVector<float>* _ei;
	SyncVector<float>* _zi;
	SyncVector<float>* _pij;
	SyncVector<float>* _eij;
	SyncVector<float>* _zj2;
	SyncVector<float>* _wij;
	SyncVector<float>* _ej;
	
};

}
}

#endif //__GSBN_PROC_STRUCTURAL_PLASTICITY_PRJ_HPP__
