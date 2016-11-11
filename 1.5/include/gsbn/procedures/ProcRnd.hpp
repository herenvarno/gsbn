#ifndef __GSBN_PROC_RND_HPP__
#define __GSBN_PROC_RND_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/SyncVector.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{

class ProcRnd : public ProcedureBase{

REGISTER(ProcRnd);

public:
	ProcRnd():_rnd(){};
	~ProcRnd(){};
	
	void init_new(NetParam net_param, Database& db);
	void init_copy(NetParam net_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	Random _rnd;
	vector<int> _mcu_start;
	vector<int> _mcu_num;
	vector<float> _snoise;
	SyncVector<float>* _rnd_uniform01;
	SyncVector<float>* _rnd_normal;
};

}


#endif // __GSBN_PROC_RND_HPP__

