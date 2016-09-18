#include "gsbn/procedures/ProcTest.hpp"

namespace gsbn{

REGISTERIMPL(ProcTest);

void ProcTest::init_new(NetParam net_param, Database& db){
	LOG(INFO) << "ProcTest init new OK!";
}
void ProcTest::init_copy(Database& db){
	LOG(INFO) << "ProcTest init copy OK!";
}

void ProcTest::update_cpu(){
	LOG(INFO) << "ProcTest update";
}

}
