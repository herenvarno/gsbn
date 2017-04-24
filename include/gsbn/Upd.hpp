#ifndef __GSBN_UPD_HPP__
#define __GSBN_UPD_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/Random.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{

/**
 * \class Upd
 * \brief The class Upd is the update engine of the Solver. Every active procedure
 * inherited from ProcedureBase and specified in configuration file will be
 * registed at run-time to this class. The initiation functions
 * of the registered procedures will be called once via init(),
 * while the update functions of the registered procedures will be called
 * periodically via update().
 */
class Upd{

public:
	/**
	 * \fn Upd
	 * \brief Just a constructor, do nothing useful.
	 */
	Upd();
	/**
	 * \fn ~Upd
	 * \brief Just a deconstructor, do nothing useful.
	 */
	~Upd();
	
	/**
	 * \fn init
	 * \brief Initialize a new Upd instance.
	 * \param solver_param The SolverParam from configuration file, used to create
	 * new procedures.
	 * \param db The Database associated with Solver.
	 * \param initialized_db The flag to indicate whether the Database has been
	 * initialized with snapshot of previous BCPNN simulation.
	 */
	void init(SolverParam solver_param, Database& db, bool initialized_db=false);
	/**
	 * \fn update
	 * \brief Periodically update each registered procedure.
	 */
	void update();

private:
	vector<ProcedureBase *> _list_proc;

};

}
#endif //_GSBN_UPD_HPP__
