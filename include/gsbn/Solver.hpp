#ifndef __GSBN_SOLVER_HPP__
#define __GSBN_SOLVER_HPP__

#include "gsbn/GlobalVar.hpp"
#include "gsbn/Upd.hpp"

namespace gsbn{

/**
 * \class Solver
 * \bref The top-level class for running a BCPNN.
 *
 * The Solver is a simulation enviroment which create everything for a BCPNN.
 * The class contains two parts: the data (Database) and the updating rule
 * (Upd). Basicly, the Solver is a machine which updates its internal data according
 * to its predefined rules.
 * There are 2 ways to create a solver, see Solver() for details.
 */
class Solver{
public:
	/**
	 * \enum type_t
	 * \bref The type of Solver.
	 */
	enum type_t{
		/** A solver created from structure description file, without initial states. */
		NEW_SOLVER,
		/** A solver created from state description file, It contains states for each
		 * modules inside the solver. The state description file is usually generated
		 * from the previous training.
		 */
		COPY_SOLVER
	};
	
	/**
	 * \fn Solver()
	 * \bref The constructor of Solver.
	 * \param type The type of solver. see Solver::type_t for more details.
	 * \param n_path The network description file.
	 * \param s_path The snapshot file of previous training. It's useful if the Solver
	 * is created in Solver::COPY_SOLVER mode.
	 */
	Solver(type_t type, string n_path, string s_path);
	
	/**
	 * \fn run()
	 * \bref The main loop of simulation.
	 */
	void run();
	
private:
	Upd _upd;
	Database _database;
	GlobalVar _glv;
};

}

#endif //__GSBN_SOLVER_HPP__
