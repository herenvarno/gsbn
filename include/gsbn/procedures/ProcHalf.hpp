#ifndef __GSBN_PROC_HALF_HPP__
#define __GSBN_PROC_HALF_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcHalf/Pop.hpp"
#include "gsbn/procedures/ProcHalf/Proj.hpp"
#include "gsbn/procedures/ProcHalf/Msg.hpp"

namespace gsbn{
namespace proc_half{

class ProcHalf : public ProcedureBase{

REGISTER(ProcHalf)

public:
	ProcHalf(){};
	~ProcHalf(){};
	
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
};

}
}

#endif // __GSBN_PROC_HALF_HPP__

