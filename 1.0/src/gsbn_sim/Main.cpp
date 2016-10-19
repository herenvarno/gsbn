#include "gsbn_sim/Main.hpp"

//#include <google/protobuf/text_format.h>
//#include <google/protobuf/io/zero_copy_stream_impl.h>
//#include <fcntl.h>

//using namespace google::protobuf;

using namespace gsbn;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
	
	set_mode(CPU);
  
	bool copy_flag = false;
	char *n_path = NULL;
	char *s_path = NULL;
	char *o_path = NULL;
	char *period = NULL;
	int p = 1;
	int c;

	while ((c = getopt (argc, argv, "n:s:o:t:")) != -1){
		switch (c){
		case 'n':
			n_path = optarg;
			break;
		case 's':
			copy_flag = true;
			s_path = optarg;
			break;
		case 'o':
			o_path = optarg;
			break;
		case 't':
			period = optarg;
			break;
		case '?':
		default:
			LOG(FATAL) << "Arguments wrong, abort!";
		}
	}
	
  for (int index = optind; index < argc; index++){
		LOG(WARNING) << "Non-option argument " << argv[index];
	}
	
	CHECK(n_path && o_path) << "Incompleted arguments!";
	
	if(period){
		sscanf(period, "%d", &p);
	}
	 
	Solver::type_t type;
	if(!copy_flag){
		type = Solver::NEW_SOLVER;
		s_path="";
	}else{
		type = Solver::COPY_SOLVER;
	}
	Solver solver(type, n_path, s_path, o_path, p);

	solver.run();
	
  return 0;
}
