#ifndef __GSBN_PROC_EXCHANGE_SPIKE_HPP__
#define __GSBN_PROC_EXCHANGE_SPIKE_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{
namespace proc_exchange_spike{


class ProcExchangeSpike: public ProcedureBase{
REGISTER(ProcExchangeSpike)

public:

	ProcExchangeSpike(){};
	~ProcExchangeSpike(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	GlobalVar _glv;
	Database* _db;
	
	int _rank;
	int _num_rank;
	MPI_Win _win;
	int _spike_buffer_size;
	
	vector<int> _shared_buffer;
	vector<int> _pop_rank;
	vector<int> _pop_num_element;
	vector<int> _pop_shared_offset;
	vector<int> _proj_src_pop;
	vector<int> _proj_dest_pop;
	vector<SyncVector<int8_t>*> _pop_sj;
	vector<SyncVector<int8_t>*> _proj_si;
};

}
}

#endif // __GSBN_PROC_EXCHANGE_SPIKE_HPP__

