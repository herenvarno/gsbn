#ifndef __GSBN_SOLVER_HPP__
#define __GSBN_SOLVER_HPP__

#include "gsbn/Gen.hpp"
#include "gsbn/Net.hpp"
#include "gsbn/Rec.hpp"

namespace gsbn{

/**
 * \class Solver
 * \bref The top-level class for running a BCPNN.
 *
 * The Solver is a simulation enviroment which create everything for a BCPNN.
 * The class provide wraps the Net, Gen as well as Rec together.
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
	 * \param i_path The description file for solver creation. It can either be
	 * a protobuf text file (structure description file) or a protobuf binary file
	 * (state description file);
	 * \param o_path The directory to dump the snapshot of solver states. It will
	 * be used by Rec class.
	 * \param period The period of 2 snapshots. Only when timestamp mod period = 0,
	 * the snapshot of solver state will be generated by Rec
	 */
	Solver(type_t type, string i_path, string o_path, int period);
	
	/**
	 * \fn run()
	 * \bref The main loop of simulation.
	 */
	void run();
	
private:
	Net _net;
	Gen _gen;
	Rec _rec;
	Database _database;
};

}

#endif //__GSBN_SOLVER_HPP__
