#include "gsbn/Solver.hpp"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fcntl.h>

using namespace google::protobuf;

namespace gsbn{

Solver::Solver(type_t type, string n_path, string s_path) : _upd(), _database(){

	SolverParam solver_param;
	int fd = open(n_path.c_str(), O_RDONLY);
	io::FileInputStream fs(fd);
	TextFormat::Parse(&fs, &solver_param);
	
	// create log directory
	string log_dir = solver_param.rec_param().directory();
	if(log_dir.empty()){
		log_dir = "./";
	}
	if(log_dir.compare(log_dir.size() - 1, 1, "/") != 0){
		log_dir = log_dir + "/";
	}
	struct stat info;
	/* Check directory exists */
	if( !(stat( log_dir.c_str(), &info ) == 0 && (info.st_mode & S_IFDIR))){
		LOG(WARNING) << "Directory does not exist! Create one!";
		string cmd="mkdir -p "+log_dir;
		if(system(cmd.c_str())!=0){
			LOG(FATAL) << "Cannot create directory for state records! Aboart!";
		}
	}
	_glv.puts("log-dir", log_dir);
	
	if(type==Solver::NEW_SOLVER){
		_database.init_new(solver_param);
		_upd.init_new(solver_param, _database);
		_glv.puti("simstep", 0);
		_glv.puti("cycle-flag", 0);
		_glv.putf("prn", 0);
		_glv.putf("old-prn", 0);
		
	}else if(type==Solver::COPY_SOLVER){
		SolverState solver_state;
		fstream input(s_path, ios::in | ios::binary);
		if (!input) {
			LOG(FATAL) << "File not found, abort!";
		} else if (!solver_state.ParseFromIstream(&input)) {
			LOG(FATAL) << "Parse file error, abort!";
		}
		
		_database.init_copy(solver_param, solver_state);
		_upd.init_copy(solver_param, _database);
		
		float dt;
		CHECK(_glv.getf("dt", dt));
		_glv.puti("simstep", int(solver_state.timestamp()/dt));
		_glv.puti("cycle-flag", 0);
		_glv.putf("prn", solver_state.prn());
		_glv.putf("old-prn", solver_state.prn());
	}else{
		LOG(FATAL) << "Unknow Solver type, abort!";
	}
}

void Solver::run(){
	clock_t start, end;
	
	float dt;
	int simstep;
	CHECK(_glv.getf("dt", dt));
	CHECK(_glv.geti("simstep", simstep));
	
	float start_time = simstep*dt;
	start = clock();
	
	int f=0;
	CHECK(_glv.geti("cycle-flag", f));
	while(f>=0){
		_upd.update();
		CHECK(_glv.geti("cycle-flag", f));
	}
	
	CHECK(_glv.geti("simstep", simstep));
	float end_time = simstep*dt;
	LOG(INFO) << "Simulation [ END ]";
	end = clock();
	
	cout << "Total time for [" << end_time - start_time << "]: "
		<< setprecision(6) << (end-start)/(double)CLOCKS_PER_SEC << endl;
}

}
