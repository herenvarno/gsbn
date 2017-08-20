#ifndef __GSBN_PROC_STRUCTURAL_PLASTICITY_POP_HPP__
#define __GSBN_PROC_STRUCTURAL_PLASTICITY_POP_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Parser.hpp"
#include "gsbn/GlobalVar.hpp"

namespace gsbn{
namespace proc_structural_plasticity{

class Pop{

public:
	Pop(int& id, int& hcu_start, int& mcu_start, PopParam pop_param, Database& db, int rank);
	~Pop();
	
	void add_prj(int prj_id);
	vector<int> get_avail_prj_list();
	vector<int> get_avail_active_mcu_list(int threshold);
	vector<int> hcu_coor(int hcu_idx);
	
	int _rank;
	int _id;
	int _dim_hcu;
	int _dim_mcu;
	int _hcu_start;
	int _mcu_start;
	
	vector<int> _shape;
	vector<int> _position;
	
	SyncVector<float>* _act;
	SyncVector<int>* _counter;
	SyncVector<int>* _fanout;
	vector<int> _avail_prj_list;
	
};

}

}

#endif //__GSBN_PROC_CHECK_POP_HPP__
