#ifndef __GSBN_PROC_SNAPSHOT_HPP__
#define __GSBN_PROC_SNAPSHOT_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{
namespace proc_snapshot{

class ProcSnapshot: public ProcedureBase{
REGISTER(ProcSnapshot)

public:	

	ProcSnapshot(){};
	~ProcSnapshot(){};
	
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
	
	GlobalVar _glv;
	Database* _db;
};

}
}

#endif // __GSBN_PROC_SNAPSHOT_HPP__

