#ifndef __GSBN_PROC_NET_GROUP_HPP__
#define __GSBN_PROC_NET_GROUP_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/procedures/ProcNetGroup/Hcu.hpp"
#include "gsbn/procedures/ProcNetGroup/Pop.hpp"
#include "gsbn/procedures/ProcNetGroup/Proj.hpp"
#include "gsbn/procedures/ProcNetGroup/Msg.hpp"
#include "gsbn/procedures/ProcNetGroup/Group.hpp"

namespace gsbn{
namespace proc_net_group{

class ProcNetGroup : public ProcedureBase{

REGISTER(ProcNetGroup)

public:
	ProcNetGroup(){};
	~ProcNetGroup(){};
	
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
	vector<Group*> _list_group;
	vector<Conn*> _list_conn;
	vector<pthread_t> _list_thread_hcu;
	vector<pthread_t> _list_thread_conn;
	Msg _msg;
	
	SyncVector<int>* _spike;
};

}
}

#endif // __GSBN_PROC_NET_HPP__

