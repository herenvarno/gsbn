#ifndef __GSBN_PROC_NET_PROJ_HPP__
#define __GSBN_PROC_NET_PROJ_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/procedures/ProcNet/Conn.hpp"
#include "gsbn/procedures/ProcNet/Hcu.hpp"
#include "gsbn/procedures/ProcNet/Pop.hpp"


namespace gsbn{
namespace proc_net{

class Proj{

public:
	Proj(){};
	~Proj(){};
	
	void init_new(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn);
	void init_copy(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, vector<Hcu*>* list_hcu, vector<Conn*>* list_conn);
	
	
	vector<Proj*>* _list_proj;
	Pop* _ptr_src_pop;
	Pop* _ptr_dest_pop;
};

}

}

#endif //__GSBN_PROC_NET_CONN_HPP__
