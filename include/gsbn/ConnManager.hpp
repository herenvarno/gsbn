#ifndef __GSBN_CONN_MANAGER_HPP__
#define __GSBN_CONN_MANAGER_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/Random.hpp"

namespace gsbn{

#define __DELAY__ 1

class ConnManager{

public:
	ConnManager();
	void init(Database& db);
	void learn(int timestamp, int stim_offset);
	void recall(int timestamp);

private:
	void update_phase_1();
	void update_phase_2(int timestamp);
	void update_phase_3();
	void update_phase_4();
	void update_phase_5();
	void update_phase_6();

	Table *_proj;
	Table *_pop;
	Table *_hcu;
	Table *_mcu;
	Table *_mcu_fanout;
	Table *_hcu_slot;
	Table *_j_array;
	Table *_i_array;
	Table *_ij_mat;
	Table *_epsc;
	Table *_wij;
	Table *_tmp1;
	Table *_tmp2;
	Table *_tmp3;
	Table *_addr;
	Table *_conn;
	Table *_conn0;
	Table *_hcu_subproj;
	
	float _kftj, _kfti;
	
	vector<int> _empty_conn0_list;
	vector<vector<int>> _existed_conn_list;
	
	float _wgain, _eps, _eps2;
	float _kp, _ke, _kzi, _kzj;
};

}

#endif //__GSBN_CONN_MANAGER_HPP__
