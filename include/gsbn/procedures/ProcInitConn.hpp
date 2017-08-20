#ifndef __GSBN_PROC_INIT_CONN_HPP__
#define __GSBN_PROC_INIT_CONN_HPP__

#include "gsbn/Random.hpp"
#include "gsbn/Database.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{
namespace proc_init_conn{

class ProcInitConn: public ProcedureBase{
REGISTER(ProcInitConn)

	class Pop{
	public:
		Pop(int& id, int& hcu_start, int& mcu_start, PopParam pop_param, Database& db, int rank);
	public:
		int _id;
		int _rank;
		int _dim_hcu;
		int _dim_mcu;
	};

	class Prj{
	public:
		Prj(int& id, vector<Pop>& pop_list, ProjParam prj_param, Database& db, int rank);
	public:
		int _id;
		int _rank;
		int _src_pop;
		int _dest_pop;
		int _dim_hcu;
		int _dim_mcu;
		int _dim_conn;
		SyncVector<int>* _ii;
		SyncVector<int>* _di;
	};

public:
	
	ProcInitConn(){};
	~ProcInitConn(){};
	
	void init_new(SolverParam solver_param, Database& db);
	void init_copy(SolverParam solver_param, Database& db);
	void update_cpu();
	#ifndef CPU_ONLY
	void update_gpu();
	#endif

private:
	void add_row(int prj_id, int src_mcu, int dest_hcu, int delay);

private:
	vector<Pop> _pop_list;
	vector<Prj> _prj_list;
};

}
}

#endif // __GSBN_PROC_INIT_CONN_HPP__

