#ifndef __GSBN_PROC_SOCK_GEN_HPP__
#define __GSBN_PROC_SOCK_GEN_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include <sys/socket.h>
#include <arpa/inet.h>

namespace gsbn{
namespace proc_sock_gen{

struct mode_t{
	int begin_step;
	int end_step;
	float prn;
	int lgidx_id;
	int lgexp_id;
	int wmask_id;
	int plasticity;
};

class ProcSockGen: public ProcedureBase{
REGISTER(ProcSockGen)

public:	

	ProcSockGen(){};
	~ProcSockGen(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	GlobalVar _glv;
	Database* _db;
	SyncVector<float>* _lginp;
	SyncVector<float>* _wmask;
	
	vector<int> _sp;
	vector<int> _mp;
	vector<int> _pop_dim_hcu;
	vector<int> _pop_dim_mcu;
	vector<int> _pop_hcu_start;
	vector<int> _pop_mcu_start;
	vector<int> _av;
	vector<int> _sv;
	vector<int> _av_cnt;
	vector<SyncVector<int8_t>*> _pop_spike;
	
	struct sockaddr_in _server;
	
	float _eps;
	string _name;
	int _prev_proc;
};

}
}

#endif // __GSBN_PROC_EXT_GEN_HPP__

