#ifndef __GSBN_PROC_FIX_MIX_HPP__
#define __GSBN_PROC_FIX_MIX_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcFixMix/Pop.hpp"
#include "gsbn/procedures/ProcFixMix/Proj.hpp"
#include "gsbn/procedures/ProcFixMix/Msg.hpp"

namespace gsbn{
namespace proc_fix_mix{

class ProcFixMix : public ProcedureBase{

REGISTER(ProcFixMix)

public:
	ProcFixMix(){};
	~ProcFixMix(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	vector<Proj*> _list_proj;
	vector<Pop*> _list_pop;
	Msg _msg;
	
	Table* _conf;
	
	int _norm_frac_bit;
	int _p_frac_bit;
};

}
}

#endif // __GSBN_PROC_FIX_HPP__

