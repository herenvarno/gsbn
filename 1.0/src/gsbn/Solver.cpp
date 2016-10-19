#include "gsbn/Solver.hpp"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fcntl.h>

using namespace google::protobuf;

namespace gsbn{

Solver::Solver(type_t type, string n_path, string s_path, string o_path, int period) : _gen(), _upd(), _rec(), _database(){
	LOG(INFO) << "111111111";
	SolverParam solver_param;
	int fd = open(n_path.c_str(), O_RDONLY);
	io::FileInputStream fs(fd);
	TextFormat::Parse(&fs, &solver_param);
		
	if(type==Solver::NEW_SOLVER){
		_database.init_new(solver_param);
		GenParam gen_param=solver_param.gen_param();
		_gen.init_new(gen_param, _database);
		NetParam net_param=solver_param.net_param();
		_upd.init_new(net_param, _database);
		_rec.init(_database);
		_rec.set_directory(o_path);
		_rec.set_period(period);
	}else if(type==Solver::COPY_SOLVER){
		SolverState solver_state;
		fstream input(s_path, ios::in | ios::binary);
    if (!input) {
      LOG(FATAL) << "File not found, abort!";
    } else if (!solver_state.ParseFromIstream(&input)) {
    	LOG(FATAL) << "Parse file error, abort!";
    }
    
    _database.init_copy(solver_param, solver_state);
		GenParam gen_param=solver_param.gen_param();
		_gen.init_copy(gen_param, _database);
		_gen.set_current_time(solver_state.timestamp());
		_gen.set_prn(solver_state.prn());
		NetParam net_param=solver_param.net_param();
		_upd.init_copy(net_param, _database);
		_rec.init(_database);
		_rec.set_directory(o_path);
		_rec.set_period(period);
    
	}else{
		LOG(FATAL) << "Unknow Solver type, abort!";
	}
}

void Solver::run(){
	clock_t start, end;
	clock_t start0, end0;
	start = clock();
	int stim=-1;
	Gen::mode_t mode=_gen.current_mode();
	float timestamp=_gen.current_time();
	float dt;
	float i=timestamp;
	if(mode!=Gen::END){
		_gen.update();
		mode=_gen.current_mode();
		timestamp=_gen.current_time();
		dt = _gen.dt();
		// Main loop
		while(mode!=Gen::END){
			switch(mode){
			case Gen::RUN:
				LOG(INFO) << "Sim[" << timestamp << "]:[ RUN ]";
				start0 = clock();
				_upd.update();
				end0 = clock();
				LOG(INFO) << "Time : " << setprecision(6) << (end0-start0)/(double)CLOCKS_PER_SEC << "";
				_rec.record();
				break;
			default:
				LOG(INFO) << "Sim[" << timestamp << "]:[ NOP ]";
				break;
			}
			_gen.update();
			mode=_gen.current_mode();
			timestamp=_gen.current_time();
		}
	}
	LOG(INFO) << "Sim[" << timestamp << "]:[ END ]";
	_rec.record(true);
	end = clock();
	LOG(INFO) << "Total time for [" << timestamp - i << "] steps: "
		<< setprecision(6) << (end-start)/(double)CLOCKS_PER_SEC << "";
}

}
