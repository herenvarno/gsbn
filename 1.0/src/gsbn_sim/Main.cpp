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
  
  /*
  SolverParam solver_param;
	int fd = open("solver.prototxt", O_RDONLY);
	io::FileInputStream fstream(fd);
	TextFormat::Parse(&fstream, &solver_param);
	
	NetParam net_param = solver_param.net_param();
	Net net;
	net.build(net_param);
	LOG(INFO) << "Table Pop:" << endl << net._pop.dump();
	LOG(INFO) << "Table Hcu:" << endl << net._hcu.dump();
	LOG(INFO) << "Table Hcu:" << endl << net._hcu.rows();
	*/
	LOG(INFO) << "ok here! 11111111";
	bool new_flag = false;
	char *i_path = NULL;
	char *o_path = NULL;
	char *period = NULL;
	int p = 1;
	int c;

	while ((c = getopt (argc, argv, "n:c:o:t:")) != -1){
		switch (c){
		case 'n':
			new_flag = true;
			i_path = optarg;
			break;
		case 'c':
			new_flag = false;
			i_path = optarg;
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
	
	CHECK(i_path && o_path) << "Incompleted arguments!";
	
	if(period){
		sscanf(period, "%d", &p);
	}
	 
	
	LOG(INFO) << "ok here! 111111122";
	Solver::type_t type;
	if(new_flag){
		type = Solver::NEW_SOLVER;
	}else{
		type = Solver::COPY_SOLVER;
	}
	LOG(INFO) << "ok here! 11111112255";
	Solver solver(type, i_path, o_path, p);
	
	LOG(INFO) << "ok here! 11111133";
	solver.run();
	
  return 0;
}