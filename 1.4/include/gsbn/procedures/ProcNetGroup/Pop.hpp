#ifndef __GSBN_PROC_NET_GROUP_POP_HPP__
#define __GSBN_PROC_NET_GROUP_POP_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcNetGroup/Conn.hpp"
#include "gsbn/procedures/ProcNetGroup/Hcu.hpp"
#include "gsbn/procedures/ProcNetGroup/Group.hpp"

namespace gsbn{
namespace proc_net_group{

class Pop{

public:
	Pop(){};
	~Pop(){};
	
	void init_new(PopParam pop_param, Database& db, vector<Pop*>* list_pop, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg);
	void init_copy(PopParam pop_param, Database& db, vector<Pop*>* list_pop, vector<Group*>* list_group, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn, Msg* msg);
	
	Hcu* get_hcu(int i);
	
	int _id;
	int _hcu_start;
	int _hcu_num;
	int _mcu_start;
	int _mcu_num;
	
	vector<Pop*>* _list_pop;
	vector<Group*>* _list_group;
	vector<Hcu*>* _list_hcu;
};

}

}

#endif //__GSBN_PROC_NET_POP_HPP__
