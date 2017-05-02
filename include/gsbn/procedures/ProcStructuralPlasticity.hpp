#ifndef __GSBN_PROC_STRUCTURAL_PLASTICITY_HPP__
#define __GSBN_PROC_STRUCTURAL_PLASTICITY_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include <algorithm>
#include <chrono>

#include "gsbn/procedures/ProcStructuralPlasticity/Coordinate.hpp"
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
	void add_row(int prj_id, int src_mcu, int dest_hcu, int delay);
	int delay_cycle(int proj, Coordinate d0, Coordinate d1);
	
	vector<Pop> _pop_list;
	vector<Prj> _prj_list;
	
	GlobalVar _glv;
	Database* _db;
	
	float _d_norm;
	float _v_cond;
	float _dt;
	int _t_th;
	int _period;
	
	vector<int> _shared_buffer;
	vector<int> _local_buffer;
	MPI_Win _win;
	
	Random _rnd;
};

}
}

#endif // __GSBN_PROC_STRUCTURAL_PLASTICITY_HPP__

