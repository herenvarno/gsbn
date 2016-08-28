#include "gsbn/Net.hpp"

namespace gsbn{

Net::Net(){
}

void Net::init(Database& db){
	_spike_manager.init(db);
	_conn_manager.init(db);
}

void Net::learn(int timestamp, int stim_offset){
	_spike_manager.learn(timestamp, stim_offset);
	_conn_manager.learn(timestamp, stim_offset);
}

void Net::recall(int timestamp){
	_spike_manager.recall(timestamp);
	_conn_manager.recall(timestamp);
}

}
