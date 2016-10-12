#ifndef __GSBN_PROC_UPD_RAND_HPP__
#define __GSBN_PROC_UPD_RAND_HPP__

#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Random.hpp"

namespace gsbn{

class ProcUpdRand : public ProcedureBase{

REGISTER(ProcUpdRand);

public:
	ProcUpdRand() : _rnd() {}
	virtual ~ProcUpdRand() {}
	
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
