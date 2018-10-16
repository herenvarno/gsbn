#ifndef __GSBN_PROC_UPD_LAZY_VS_NOCOL_HPP__
#define __GSBN_PROC_UPD_LAZY_VS_NOCOL_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcUpdLazyVsNocol/Pop.hpp"
#include "gsbn/procedures/ProcUpdLazyVsNocol/Proj.hpp"

namespace gsbn
{
namespace proc_upd_lazy_vs_nocol
{

class ProcUpdLazyVsNocol : public ProcedureBase
{

	REGISTER(ProcUpdLazyVsNocol)

public:
	ProcUpdLazyVsNocol(){};
	~ProcUpdLazyVsNocol(){};

	void init_new(SolverParam solver_param, Database &db);
	void init_copy(SolverParam solver_param, Database &db);
	void update_cpu();
#ifndef CPU_ONLY
	void update_gpu();
#endif

private:
	vector<Proj *> _list_proj;
	vector<Pop *> _list_pop;
	GlobalVar _glv;
};

} // namespace proc_upd_lazy_vs_nocol
} // namespace gsbn

#endif // __GSBN_PROC_UPD_LAZY_VS_NOCOL_HPP__
