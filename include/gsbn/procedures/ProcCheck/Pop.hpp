#ifndef __GSBN_PROC_CHECK_POP_HPP__
#define __GSBN_PROC_CHECK_POP_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Parser.hpp"
#include "gsbn/GlobalVar.hpp"

namespace gsbn{
namespace proc_check{

class Pop{

public:
	Pop(int& id, int& hcu_start, int& mcu_start, PopParam pop_param, Database& db);
	~Pop();
	
	void update_counter(int cursor);
	void check_result();
	void send_result();
	void set_win_idx(MPI_Win win_idx);
	void set_win_cnt(MPI_Win win_cnt);
	
	int _rank;
	int _id;
	int _dim_hcu;
	int _dim_mcu;
	int _hcu_start;
	int _mcu_start;
	
	MPI_Win _win_idx;
	MPI_Win _win_cnt;
	
	SyncVector<int8_t>* _spike;
	vector<vector<int>> _counter;
	vector<int> _active_mcu_idx;
	vector<int> _active_mcu_cnt;
	
};

}

}

#endif //__GSBN_PROC_CHECK_POP_HPP__
