#ifndef __GSBN_PROC_NET_BATCH_HPP__
#define __GSBN_PROC_NET_BATCH_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcNetBatch/Pop.hpp"
#include "gsbn/procedures/ProcNetBatch/Proj.hpp"
#include "gsbn/procedures/ProcNetBatch/Msg.hpp"

namespace gsbn{
namespace proc_net_batch{

class ProcNetBatch : public ProcedureBase{

REGISTER(ProcNetBatch)

public:
	ProcNetBatch(){};
	~ProcNetBatch(){};
	
	void init_new(NetParam net_param, Database& db);
	void init_copy(NetParam net_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	vector<Proj*> _list_proj;
	vector<Pop*> _list_pop;
	Msg _msg;
	
	SyncVector<int>* _spike;
};

}
}

#endif // __GSBN_PROC_NET_HPP__

