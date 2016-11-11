#ifndef __GSBN_PROC_NET_HPP__
#define __GSBN_PROC_NET_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcNet/Hcu.hpp"
#include "gsbn/procedures/ProcNet/Pop.hpp"
#include "gsbn/procedures/ProcNet/Proj.hpp"
#include "gsbn/procedures/ProcNet/Msg.hpp"

namespace gsbn{
namespace proc_net{

class ProcNet : public ProcedureBase{

REGISTER(ProcNet);

public:
	ProcNet(){};
	~ProcNet(){};
	
	void init_new(NetParam net_param, Database& db);
	void init_copy(NetParam net_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	vector<Proj*> _list_proj;
	vector<Pop*> _list_pop;
	vector<Hcu*> _list_hcu;
	vector<Conn*> _list_conn;
	vector<pthread_t> _list_thread_hcu;
	vector<pthread_t> _list_thread_conn;
	Msg _msg;
	
	SyncVector<int>* _spike;
};

}
}

#endif // __GSBN_PROC_NET_HPP__

