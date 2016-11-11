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
		RUN,
		END
	};
	
	/**
	 * \fn Gen()
	 * \bref The constructor of Gen.
	 */
	Gen();
	
	/**
	 * \fn init()
	 * \bref Initialize the Gen
	 *
	 * Gen object cannot be used before init() function is called.
	 */
	void init_new(GenParam gen_param, Database& db);
	void init_copy(GenParam gen_param, Database& db);
	/**
	 * \fn update()
	 * \bref Update the state of Gen
	 *
	 * The function increase the simulation time, and it extracts the simulation
	 * mode and proper stimulus. It will update "conf" table and store all the
	 * information in it.
	 */
	void update();
	
	/**
	 * \fn set_current_time()
	 * \bref Manually set the simulation time.
	 */
	void set_current_time(float timestamp);
	/**
	 * \fn current_time()
	 * \bref Get the simulation time.
	 * \return The simulation timestamp.
	 */
	float current_time();
	/**
	 * \fn dt()
	 * \bref Get the delta time for one step.
	 * \return The dt.
	 */
	float dt();
	/**
	 * \fn current_mode()
	 * \bref Get the simulation mode.
	 * \return The simulation mode.
	 */
	mode_t current_mode();
	/**
	 * \fn set_max_time()
	 * \bref Manually set the max simulation time.
	 */
	void set_max_time(float time);
	/**
	 * \fn set_dt()
	 * \bref Manually set the dt.
	 */
	void set_dt(float time);
	
	void set_prn(float prn);

private:
	
	Table *_mode;
	Table *_conf;
	
	SyncVector<float>* _lginp;
	SyncVector<float>* _wmask;
	
	int _current_step;
	float _max_time;
	mode_t _current_mode;
	float _dt;
	int _cursor;
	
	// deprecated
	Table *_stim;

	float _current_time;
};

}

#endif //__GSBN_GEN_HPP__
