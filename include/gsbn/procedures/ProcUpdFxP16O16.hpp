#ifndef __GSBN_PROC_UPD_FX_P16_O16_HPP__
#define __GSBN_PROC_UPD_FX_P16_O16_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcUpdFxP16O16/Pop.hpp"
#include "gsbn/procedures/ProcUpdFxP16O16/Proj.hpp"
#include "gsbn/procedures/ProcUpdFxP16O16/Msg.hpp"

namespace gsbn{
namespace proc_upd_fx_p16_o16{

class ProcUpdFxP16O16 : public ProcedureBase{

REGISTER(ProcUpdFxP16O16)

public:
	ProcUpdFxP16O16(){};
	~ProcUpdFxP16O16(){};
	
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

#endif // __GSBN_PROC_UPD_FX_P16_O16_HPP__

