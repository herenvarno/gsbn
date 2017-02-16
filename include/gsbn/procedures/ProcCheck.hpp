#ifndef __GSBN_PROC_CHECK_HPP__
#define __GSBN_PROC_CHECK_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{
namespace proc_check{

struct mode_t{
	int begin_step;
	int end_step;
	float prn;
	int lgidx_id;
	int lgexp_id;
	int wmask_id;
	int plasticity;
};

class ProcCheck: public ProcedureBase{
REGISTER(ProcCheck)

public:	

	ProcCheck(){};
	~ProcCheck(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
//	Table *_mode;
	Table *_conf;
	vector<mode_t> _list_mode;
	SyncVector<float>* _lginp;
	SyncVector<float>* _wmask;
	SyncVector<int32_t>* _lgidx;
	SyncVector<int32_t>* _count;
	
	vector<int> _mcu_in_hcu;
	vector<int> _mcu_in_pop;
	
	int _pattern_num;
	int _correct_pattern_num;
	
	int _threashold;
	string _logfile;
	
	Database *_db;
	int _cursor;
	
	int _spike_buffer_cursor;
	int _spike_buffer_size;
};

}
}

#endif // __GSBN_PROC_CHECK_HPP__

