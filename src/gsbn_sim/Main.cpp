#include "gsbn_sim/Main.hpp"

//#include <google/protobuf/text_format.h>
//#include <google/protobuf/io/zero_copy_stream_impl.h>
//#include <fcntl.h>

//using namespace google::protobuf;

using namespace gsbn;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
	
	bool copy_flag = false;
	char *n_path = NULL;
	char *s_path = NULL;
	char *m_mode = NULL;
	bool log_flag = false;
	int p = 1;
	int c;

	while ((c = getopt (argc, argv, "n:s:m:l")) != -1){
		switch (c){
		case 'n':
			n_path = optarg;
			break;
		case 's':
			copy_flag = true;
			s_path = optarg;
			break;
		case 'm':
			m_mode = optarg;
			break;
		case 'l':
			log_flag = true;
			break;
		case '?':
		default:
			LOG(FATAL) << "Arguments wrong, abort!";
		}
	}
	
  for (int index = optind; index < argc; index++){
		LOG(WARNING) << "Non-option argument " << argv[index];
	}
	
	CHECK(n_path) << "Incompleted arguments!";
	
	if(m_mode){
		if(strcmp(m_mode,"CPU")==0 || strcmp(m_mode,"cpu")==0){
			set_mode(CPU);
		}else if(strcmp(m_mode,"GPU")==0 || strcmp(m_mode,"gpu")==0){
			set_mode(GPU);
		}else{
			LOG(FATAL)<<"Wrong mode!";
		}
	}else{
		set_mode(CPU);
	}
	
	if(!log_flag){
		el::Configurations cc;
		cc.setToDefault();
		cc.parseFromText("*GLOBAL:\n ENABLED = false");
		el::Loggers::reconfigureAllLoggers(cc);
	}
	
	Solver::type_t type;
	if(!copy_flag){
		type = Solver::NEW_SOLVER;
		s_path="";
	}else{
		type = Solver::COPY_SOLVER;
	}
	Solver solver(type, n_path, s_path);

	solver.run();
	
  return 0;
}
