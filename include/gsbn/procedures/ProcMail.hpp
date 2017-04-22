#ifndef __GSBN_PROC_MAIL_HPP__
#define __GSBN_PROC_MAIL_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include <algorithm>
#include <chrono>

#include "gsbn/procedures/ProcMail/Msg.hpp"
#include "gsbn/procedures/ProcMail/Coordinate.hpp"

namespace gsbn{
namespace proc_mail{

class ProcMail: public ProcedureBase{
REGISTER(ProcMail)

public:	

	ProcMail(){};
	~ProcMail(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	void send_spike();
	void receive_spike();
	void add_row(int proj, int src_mcu, int dest_hcu, int delay);
	void update_avail_hcu(int pop, int src_mcu, int proj_id, int dest_hcu, bool remove_all);
	bool validate_conn(int pop, int src_mcu, int proj_id, int dest_hcu);
	void init_proj_conn(int proj, int init_conn_rate);
	int delay_cycle(int proj, Coordinate d0, Coordinate d1);
	
	vector<int> _avail_mails;
	
	vector<int> _pop_rank;
	vector<vector<int>> _pop_shape;
	vector<int> _pop_dim_hcu;
	vector<int> _pop_dim_mcu;
	vector<int> _pop_hcu_start;
	vector<int> _pop_mcu_start;
	int _spike_buffer_size;
	vector<vector<int>> _pop_avail_proj;
	vector<vector<int>> _pop_avail_proj_hcu_start;
	vector<int> _pop_shared_buffer_start;
	vector<vector<vector<vector<int>>>> _pop_avail_hcu;
	
	vector<int> _proj_src_pop;
	vector<int> _proj_dest_pop;
	vector<int> _proj_dim_conn;
	vector<float> _proj_distance;
	vector<vector<int>> _proj_conn_cnt;
	vector<int> _proj_shared_buffer_start;
	vector<int> _proj_local_buffer_start;
	
	vector<int> _tmp_src;
	vector<int> _tmp_dest;
	vector<int> _tmp_proj;
	
	vector<SyncVector<int>*> _pop_fanout;
	vector<SyncVector<int8_t>*> _pop_spike;
	vector<SyncVector<int> *> _proj_slot;
	vector<SyncVector<int> *> _proj_ii;
	vector<SyncVector<int> *> _proj_di;
	
	Msg _msg;
	
	GlobalVar _glv;
	Database* _db;
	
	float _d_norm;
	float _v_cond;
	float _dt;
	
	vector<int> _shared_buffer;
	vector<int> _local_buffer;
	MPI_Win _win;
	
	Random _rnd;
};

}
}

#endif // __GSBN_PROC_MAIL_HPP__

