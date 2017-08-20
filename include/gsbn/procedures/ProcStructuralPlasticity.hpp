#ifndef __GSBN_PROC_STRUCTURAL_PLASTICITY_HPP__
#define __GSBN_PROC_STRUCTURAL_PLASTICITY_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include <algorithm>
#include <chrono>

#include "gsbn/procedures/ProcStructuralPlasticity/Pop.hpp"
#include "gsbn/procedures/ProcStructuralPlasticity/Prj.hpp"

namespace gsbn{
namespace proc_structural_plasticity{

class ProcStructuralPlasticity: public ProcedureBase{
REGISTER(ProcStructuralPlasticity)

public:
	
	ProcStructuralPlasticity(){};
	~ProcStructuralPlasticity(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	void init_conn();

private:
	void add_row(int prj_id, int src_mcu, int dest_hcu, int delay);
	int delay_cycle(int prj_idx, int src_mcu, int dest_hcu);
	
	vector<Pop> _pop_list;
	vector<Prj> _prj_list;
	
	GlobalVar _glv;
	Database* _db;
	
	float _d_norm;
	float _v_cond;
	float _dt;
	int _t_th;
	int _period;
	int _pruning_period;
	int _enable_geometry;
	int _enable_init_conn;
	float _wp;
	float _wn;
	
	vector<int> _shared_buffer;
	vector<int> _local_buffer;
	MPI_Win _win;
	
	Random _rnd;
};

}
}

#endif // __GSBN_PROC_STRUCTURAL_PLASTICITY_HPP__

