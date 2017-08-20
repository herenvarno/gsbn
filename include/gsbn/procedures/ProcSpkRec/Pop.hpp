#ifndef __GSBN_PROC_SPK_REC_POP_HPP__
#define __GSBN_PROC_SPK_REC_POP_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/Parser.hpp"
#include "gsbn/GlobalVar.hpp"

namespace gsbn{
namespace proc_spk_rec{

class Pop{

public:
	Pop(int& id, PopParam pop_param, Database& db);
	~Pop();
	
	void record(string filename, int simstep);

	int _rank;
	int _id;
	int _dim_hcu;
	int _dim_mcu;
	int _spike_buffer_size;
	int _maxfq;
	
	SyncVector<int8_t>* _spike;
};

}

}

#endif //__GSBN_PROC_SPK_REC_POP_HPP__
