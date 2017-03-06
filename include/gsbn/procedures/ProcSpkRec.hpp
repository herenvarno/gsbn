#ifndef __GSBN_PROC_SPK_REC_HPP__
#define __GSBN_PROC_SPK_REC_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{
namespace proc_spk_rec{

class ProcSpkRec: public ProcedureBase{
REGISTER(ProcSpkRec)

public:	

	ProcSpkRec(){};
	~ProcSpkRec(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	string _directory;
	int _offset;
	int _period;
	int _spike_buffer_size;
	
	vector<SyncVector<int8_t> *> _spikes;
	
	GlobalVar _glv;
	Database* _db;
};

}
}

#endif // __GSBN_PROC_SPK_REC_HPP__

