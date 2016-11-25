#ifndef __GSBN_PROC_FIX_HPP__
#define __GSBN_PROC_FIX_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcFix/Pop.hpp"
#include "gsbn/procedures/ProcFix/Proj.hpp"
#include "gsbn/procedures/ProcFix/Msg.hpp"

namespace gsbn{
namespace proc_fix{

class ProcFix : public ProcedureBase{

REGISTER(ProcFix)

public:
	ProcFix(){};
	~ProcFix(){};
	
	void init_new(NetParam net_param, Database& db);
	void init_copy(NetParam net_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	vector<Proj*> _list_proj;
	vector<Pop*> _list_pop;
	Msg _msg;
};

}
}

#endif // __GSBN_PROC_FIX_HPP__

