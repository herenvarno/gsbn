#ifndef __GSBN_UPD_HPP__
#define __GSBN_UPD_HPP__

#include "gsbn/Database.hpp"
#include "gsbn/Random.hpp"
#include "gsbn/ProcedureFactory.hpp"

namespace gsbn{
class Upd{

public:
	Upd();
	
	void init(Database& db);
	void init_new(NetParam, Database& db);
	void init_copy(NetParam, Database& db);
	void update();

private:
	vector<ProcedureBase *> _list_proc;

};

}
#endif //_GSBN_UPD_HPP__
