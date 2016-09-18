#ifndef __GSBN_PROC_TEST_HPP__
#define __GSBN_PROC_TEST_HPP__

#include "gsbn/ProcedureFactory.hpp"
#include "gsbn/Database.hpp"

namespace gsbn{

class ProcTest : public ProcedureBase{

REGISTER(ProcTest);

public:
	virtual ~ProcTest() {}
	
	void init_new(NetParam net_param, Database& db);
	void init_copy(Database& db);
	void update_cpu();

};

}

#endif
