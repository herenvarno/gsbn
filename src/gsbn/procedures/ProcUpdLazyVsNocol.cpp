#include "gsbn/procedures/ProcUpdLazyVsNocol.hpp"

namespace gsbn
{
namespace proc_upd_lazy_vs_nocol
{

REGISTERIMPL(ProcUpdLazyVsNocol);

void ProcUpdLazyVsNocol::init_new(SolverParam solver_param, Database &db)
{
	NetParam net_param = solver_param.net_param();

	ProcParam proc_param = get_proc_param(solver_param);

#ifndef CPU_ONLY
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	int gpu_device_id = 0;
#endif

	int hcu_cnt = 0;
	int mcu_cnt = 0;
	int pop_param_size = net_param.pop_param_size();
	for (int i = 0; i < pop_param_size; i++)
	{
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for (int j = 0; j < pop_num; j++)
		{
			Pop *p = new Pop();
			p->init_new(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt);
		}
	}

#ifndef CPU_ONLY
	gpu_device_id = 0;
#endif

	int total_pop_num = _list_pop.size();
	int proj_param_size = net_param.proj_param_size();
	for (int i = 0; i < proj_param_size; i++)
	{
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if (src_pop < total_pop_num && dest_pop < total_pop_num)
		{
			Proj *proj = new Proj();
			proj->init_new(proc_param, proj_param, db, &_list_proj, &_list_pop);
		}
	}
}

void ProcUpdLazyVsNocol::init_copy(SolverParam solver_param, Database &db)
{
	NetParam net_param = solver_param.net_param();

	ProcParam proc_param = get_proc_param(solver_param);

#ifndef CPU_ONLY
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	int gpu_device_id = 0;
#endif

	int hcu_cnt = 0;
	int mcu_cnt = 0;
	int pop_param_size = net_param.pop_param_size();
	for (int i = 0; i < pop_param_size; i++)
	{
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for (int j = 0; j < pop_num; j++)
		{
			Pop *p = new Pop();
			p->init_copy(proc_param, pop_param, db, &_list_pop, &hcu_cnt, &mcu_cnt);
		}
	}

#ifndef CPU_ONLY
	gpu_device_id = 0;
#endif

	int total_pop_num = _list_pop.size();
	int proj_param_size = net_param.proj_param_size();
	for (int i = 0; i < proj_param_size; i++)
	{
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if (src_pop < total_pop_num && dest_pop < total_pop_num)
		{
			Proj *proj = new Proj();
			proj->init_copy(proc_param, proj_param, db, &_list_proj, &_list_pop);
		}
	}
}

void ProcUpdLazyVsNocol::update_cpu()
{
	int cycle_flag;
	CHECK(_glv.geti("cycle-flag", cycle_flag));
	if (cycle_flag != 1)
	{
		return;
	}
	int rank;
	CHECK(_glv.geti("rank", rank));

	int simstep;
	float dt;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.getf("dt", dt));
	if (simstep % (int(1 / dt)) == 0)
	{
		LOG(INFO) << "Sim [ " << simstep * dt << " ]";
	}
	for (auto it = _list_pop.begin(); it != _list_pop.end(); it++)
	{
		if (rank != (*it)->_rank)
		{
			continue;
		}
		(*it)->update_rnd_cpu();
		(*it)->update_sup_cpu();
		(*it)->fill_spike();
	}
	for (auto it = _list_proj.begin(); it != _list_proj.end(); it++)
	{
		if (rank != (*it)->_rank)
		{
			continue;
		}
		(*it)->update_all_cpu();
		(*it)->update_jxx_cpu();
		(*it)->update_ssi_cpu();
		(*it)->update_ssj_cpu();
		(*it)->update_row_cpu();
		(*it)->update_col_cpu();
		(*it)->update_que_cpu();
	}
}

#ifndef CPU_ONLY

void ProcUpdLazyVsNocol::update_gpu()
{
	update_cpu();
}

#endif

} // namespace proc_upd_lazy_vs_nocol
} // namespace gsbn
