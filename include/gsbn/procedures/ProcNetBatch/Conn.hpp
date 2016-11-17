#ifndef __GSBN_PROC_NET_BATCH_HPP__
#define __GSBN_PROC_NET_BATCH_HPP__

#include "gsbn/Database.hpp"

namespace gsbn{
namespace proc_net_batch{

class Conn{
public:
	Conn(){};
	~Conn(){};
	
	void add_row(int src_mcu, int dest_hcu, int delay);
	
public:
	int _proj_id;
	vector<int> *_avail_hcu;
	vector<int> *_conn_cnt;
	
	SyncVector<int> *_ii;
	SyncVector<int> *_di;
};

}
}

#endif
