#ifndef __GSBN_PROC_SPK_REC_HPP__
#define __GSBN_PROC_SPK_REC_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcSpkRec/Pop.hpp"
#include "gsbn/procedures/ProcSpkRec/Prj.hpp"

namespace gsbn
{
namespace proc_spk_rec
{

/**
 * \class ProcSpkRec
 * \brief The class ProcSpkRec is a procedure which records spikes for each
 * population.
 *
 * ## Configuration
 * All configuration parameters are optional and should be specified in
 * if necessary configuration file.
 * - offset : (unsigned int, default=0) The simulation step which starts the
 * recording process.
 * - period : (unsigned int, default=1) The number of simulation step between two
 * recording process.
 * 
 * ## Recording file
 * The recording file is located in directory $REC_ROOT/ProcSpkRec . The directory
 * will be created automatically if it doesn't exist before.
 * 
 * Each recording file records spikes for one population. The file is in CSV
 * format.
 *
 * The first line has 4 fields:
 * - number of HCU in this population
 * - number of MCU in each HCU
 * - Time difference between 2 simulation step (dt). Typical value is 0.001(=1ms).
 * - Maximum firing frequency. Typical value is 100(=100Hz).
 *
 * The following lines have at least 2 fields:
 * - timestamp
 * - the active MCU index. Index range is 0 to total amount of this population.
 */
class ProcSpkRec : public ProcedureBase
{
	REGISTER(ProcSpkRec)

public:
	/**
	 * \fn ProcSpkRec
	 * \brief Just a constructor, do nothing useful.
	 */
	ProcSpkRec(){};
	/**
	 * \fn ~ProcSpkRec
	 * \brief Just a deconstructor, do nothing useful.
	 */
	~ProcSpkRec(){};

	/**
	 * \fn init_new
	 * \brief Initialize a new instance.
	 * \param solver_param The SolverParam from configuration file, used to create
	 * new procedures.
	 * \param db The Database associated with Solver.
	 */
	void init_new(SolverParam solver_param, Database &db);
	/**
	 * \fn init_copy
	 * \brief Initialize a new instance from previous snapshot.
	 * \param solver_param The SolverParam from configuration file, used to create
	 * new procedures.
	 * \param db The Database associated with Solver.
	 */
	void init_copy(SolverParam solver_param, Database &db);
	/**
	 * \fn update_cpu
	 * \brief Periodical update function for CPU mode
	 */
	void update_cpu();
#ifndef CPU_ONLY
	/**
	 * \fn update_gpu
	 * \brief Periodical update function for GPU mode
	 */
	void update_gpu();
#endif

private:
	string _directory;
	int _offset;
	int _period;

	int _rank;
	vector<Pop> _pop_list;
	vector<Prj> _prj_list;

	GlobalVar _glv;
	Database *_db;
};

} // namespace proc_spk_rec
} // namespace gsbn

#endif // __GSBN_PROC_SPK_REC_HPP__
