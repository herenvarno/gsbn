#ifndef __GSBN_NET_HPP__
#define __GSBN_NET_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/SpikeManager.hpp"
#include "gsbn/ConnManager.hpp"

namespace gsbn{

class Net{

public:
	Net();
	
	void init(Database& db);
	void learn(int timestamp, int stim_offset);
	void recall(int timestamp);

private:
	SpikeManager _spike_manager;
	ConnManager _conn_manager;
};

}
#endif //_GSBN_NET_HPP__
