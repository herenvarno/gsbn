#ifndef __GSBN_PROC_UPD_PERIODIC_HPP__
#define __GSBN_PROC_UPD_PERIODIC_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcUpdPeriodic/Pop.hpp"
#include "gsbn/procedures/ProcUpdPeriodic/Proj.hpp"
#include "gsbn/procedures/ProcUpdPeriodic/Msg.hpp"

namespace gsbn{
namespace proc_upd_periodic{

/**
 * \class ProcUpdPeriodic
 * \bref Main update procedure of BCPNN, All traces are implemented in floating-point
 * type. All the traces are represented by fp32 (float).
 *
 * ## Configuration
 * The configuration parameter are listed below. They should be setted in network description file.
 * - argf[0] : (float, optional, default=0) Initial connection rate, range [0, 1].
 * - args[0] : (string, optional) External spikes file. If this value is set, the network will replace
 * the output spike to the external spikes according to this file. It is used for
 * debug purpose.
 */
class ProcUpdPeriodic : public ProcedureBase{

REGISTER(ProcUpdPeriodic)

public:
	ProcUpdPeriodic(){};
	~ProcUpdPeriodic(){};
	
	/**
	 * \fn init_new
	 * \bref Initialize an object from a new solver parameter. This function will
	 * be called only once at the initilization stage.
	 * \param solver_param The solver parameter that defines the network.
	 * \param db The internal database class instance.
	 */
	void init_new(SolverParam solver_param, Database& db);
	/**
	 * \fn init_copy
	 * \bref Initialize an object from a solver parameter which stores the data from
	 * previous snapshot. This function will be called only once at the
	 * initilization stage.
	 * \param solver_param The solver parameter that defines the network.
	 * \param db The internal database class instance.
	 */
	void init_copy(SolverParam solver_param, Database& db);
	/**
	 * \fn update_cpu
	 * \bref Update the internal state of the class. All the operation will be done
	 * in main memory by CPU. This function will be called in every update cycle.
	 */
	void update_cpu();
	#ifndef CPU_ONLY
	/**
	 * \fn update_gpu
	 * \bref Update the internal state of the class. Some the operation will be done
	 * in GPU memory by GPU. This function will be called in every update cycle.
	 */
	void update_gpu();
	#endif

private:
	vector<Proj*> _list_proj;
	vector<Pop*> _list_pop;
	Msg _msg;
	Table* _conf;
};

}
}

#endif // __GSBN_PROC_UPD_PERIODIC_HPP__

