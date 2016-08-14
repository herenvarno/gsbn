#include "gsbn/Main.hpp"

using namespace gsbn;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
	Random::init();
	for(int i=0; i<10; i++){
	LOG(INFO) << Random::gen_normal(5, 300);
	}
	
	return 0;
}
