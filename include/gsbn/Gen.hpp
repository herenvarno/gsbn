#ifndef __GSBN_GEN_HPP__
#define __GSBN_GEN_HPP__

#include "gsbn/Database.hpp"

namespace gsbn{

/**
 * \class Gen
 * \bref The generator of the BCPNN simulation environment.
 *
 * The Gen manage the simulation time, simulation mode and the simuli. It works
 * as the generator of the Solver.
 */
class Gen{

public:
	
	enum mode_t{
		NOP,
		LEARN,
		RECALL,
		END
	};
	
	/**
	 * \fn Gen()
	 * \bref The constructor of Gen.
	 */
	Gen();
	
	
	void init(Database& db);
	/**
	 * \fn update()
	 * \bref Update the state of Gen
	 *
	 * The function increase the simulation time, and it extracts the simulation
	 * mode and proper stimulus.
	 */
	void update();
	
	/**
	 * \fn set_current_time()
	 * \bref Manually set the simulation time.
	 */
	void set_current_time(int timestamp);
	/**
	 * \fn current_time()
	 * \bref Get the simulation time.
	 * \return The simulation timestamp.
	 */
	int current_time();
	/**
	 * \fn current_mode()
	 * \bref Get the simulation mode.
	 * \return The simulation mode.
	 */
	mode_t current_mode();
	/**
	 * \fn current_stim()
	 * \bref Get the simulation stimulus.
	 * \return The index of simulation stimulus.
	 */
	int current_stim();
	
	void set_max_time(int time);

private:
	
	Table *_mode;
	Table *_stim;
	
	int _current_time;
	mode_t _current_mode;
	int _current_stim;
	int _max_time;
};

}

#endif //__GSBN_GEN_HPP__
