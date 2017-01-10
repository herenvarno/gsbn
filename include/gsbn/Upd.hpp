#ifndef __GSBN_UPD_HPP__
#define __GSBN_UPD_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/Random.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{

/**
 * \class Upd
 * \bref The class Upd is the update engine of the Solver. Every active procedure
 * inherited from ProcedureBase will be registed to this class. The initiation functions
 * of the registered procedures will be called once, while the update functions of
 * the registered procedures will be called at every update cycle of Solver.
 */
class Upd{

public:
	Upd();
	
	void init(Database& db);
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update();

private:
	vector<ProcedureBase *> _list_proc;

};

}
#endif //_GSBN_UPD_HPP__
