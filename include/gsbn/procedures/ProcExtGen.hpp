#ifndef __GSBN_PROC_EXT_GEN_HPP__
#define __GSBN_PROC_EXT_GEN_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{
namespace proc_ext_gen{

struct mode_t{
	int begin_step;
	int end_step;
	float prn;
	int lgidx_id;
	int lgexp_id;
	int wmask_id;
	int plasticity;
};

class ProcExtGen: public ProcedureBase{
REGISTER(ProcExtGen)

public:	

	ProcExtGen(){};
	~ProcExtGen(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	GlobalVar _glv;
	vector<mode_t> _list_mode;
	SyncVector<int>* _lgidx;
	SyncVector<float>* _lginp;
	SyncVector<float>* _wmask;
	
	vector<int> _mcu_in_hcu;
	
	int _old_lgidx_id;
	float _eps;
	
	int _cursor;
};

}
}

#endif // __GSBN_PROC_EXT_GEN_HPP__

