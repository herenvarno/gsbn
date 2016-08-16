#ifndef __GSBN_NET_HPP__
#define __GSBN_NET_HPP__

namespace gsbn{

class Net{

public:
	Net();
	set_state();
	
	update();
	
	set_learn();
	tables();

private:
	bool _learn;
	Pop pop;
	Hcu hcu;
	HcuSlot hcu_slot;
	McuFanout mcu_fanout;
	Spike spike;
	IArray i_array;
	JArray j_array;
	IJMat ij_mat;
	Proj proj;
};

}

#endif //__GSBN_NET_HPP__
