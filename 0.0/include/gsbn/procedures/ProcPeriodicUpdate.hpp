#ifndef __GSBN_PROC_PERIODIC_UPDATE_HPP__
#define __GSBN_PROC_PERIODIC_UPDATE_HPP__

#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Random.hpp"
#include "cblas.h"

namespace gsbn{

class ProcPeriodicUpdate : public ProcedureBase{

REGISTER(ProcPeriodicUpdate);

public:
	ProcPeriodicUpdate() : _rnd() {}
	virtual ~ProcPeriodicUpdate() {}
	
	void init_new(NetParam net_param, Database& db);
	void init_copy(Database& db);
	void update_cpu();
	#ifndef GPU_ONLY
	void update_gpu();
	#endif

private:
	Random _rnd;
	
	Table *_hcu;
	Table *_mcu;
	Table *_rnd_uniform01;
	Table *_rnd_normal;

};

}

#endif
