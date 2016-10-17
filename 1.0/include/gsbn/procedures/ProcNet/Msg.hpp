#ifndef __GSBN_PROC_NET_MSG_HPP__
#define __GSBN_PROC_NET_MSG_HPP__

#include "gsbn/Database.hpp"

namespace gsbn{
namespace proc_net{

struct msg_t{
	int src_hcu;
	int src_mcu;
	int dest_hcu;
	int dest_mcu;
	int delay;
	int type;
};

class Msg{
public:
	void init_new(NetParam net_param, Database& db);
	void init_copy(NetParam net_param, Database& db);
	void update();
	void send(int src_hcu, int src_mcu, int dest_hcu, int dest_mcu, int type);
	vector<msg_t> receive(int hcu_id);
	void clear_empty_pos();
	int calc_delay(int src_hcu, int dest_hcu);
	
private:
	SyncVector<int> _msgbox;
	vector<int> _empty_pos;
	vector<vector<msg_t>> _list_active_msg;
};


}
}

#endif // __GSBN_PROC_NET_MSG_HPP__
