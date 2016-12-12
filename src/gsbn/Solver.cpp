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
	
	CHECK(_conf=_database.create_table(".conf", {
		sizeof(int), sizeof(float), sizeof(float), sizeof(float),
		sizeof(int), sizeof(int), sizeof(int), sizeof(int)
	}));
	_conf->expand(1);
	if(type==Solver::NEW_SOLVER){
		_database.init_new(solver_param);
		_upd.init_new(solver_param, _database);
		
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
		int* ptr_conf0 = static_cast<int*>(_conf->mutable_cpu_data(0));
		float* ptr_conf1 = static_cast<float*>(_conf->mutable_cpu_data(0));
		float dt = ptr_conf1[Database::IDX_CONF_DT];
		ptr_conf0[Database::IDX_CONF_TIMESTAMP]=int(solver_state.timestamp()/dt);
		ptr_conf1[Database::IDX_CONF_PRN]=solver_state.prn();
	}else{
		LOG(FATAL) << "Unknow Solver type, abort!";
	}
}

void Solver::run(){
	clock_t start, end;
	
	const int* ptr_conf0 = static_cast<const int*>(_conf->cpu_data(0));
	const float* ptr_conf1 = static_cast<const float*>(_conf->cpu_data(0));
	float dt = ptr_conf1[Database::IDX_CONF_DT];
	float start_time = ptr_conf0[Database::IDX_CONF_TIMESTAMP]*dt;
	start = clock();
	while(ptr_conf0[Database::IDX_CONF_MODE]>=0){
		_upd.update();
	}
	float end_time = ptr_conf0[Database::IDX_CONF_TIMESTAMP]*dt;
	LOG(INFO) << "Simulation [ END ]";
	end = clock();
	
	cout << "Total time for [" << end_time - start_time << "]: "
		<< setprecision(6) << (end-start)/(double)CLOCKS_PER_SEC << endl;
}

}
