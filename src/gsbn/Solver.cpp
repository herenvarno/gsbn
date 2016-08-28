#include "gsbn/Solver.hpp"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fcntl.h>

using namespace google::protobuf;

namespace gsbn{

Solver::Solver(type_t type, string i_path, string o_path, int period) : _gen(), _net(), _rec(), _database(){
	if(type==Solver::NEW_SOLVER){
		SolverParam solver_param;
		int fd = open(i_path.c_str(), O_RDONLY);
		io::FileInputStream fstream(fd);
		TextFormat::Parse(&fstream, &solver_param);
		_database.init_new(solver_param);
		
	}else if(type==Solver::COPY_SOLVER){
		SolverState solver_state;
		fstream input(i_path, ios::in | ios::binary);
    if (!input) {
      LOG(FATAL) << "File not found, abort!";
    } else if (!solver_state.ParseFromIstream(&input)) {
    	LOG(FATAL) << "Parse file error, abort!";
    }
    _database.init_copy(solver_state);
    _gen.set_current_time(solver_state.timestamp());
    
	}else{
		LOG(FATAL) << "Unknow Solver type, abort!";
	}
	// gen
	_gen.init(_database);
	
	// net
	_net.init(_database);
	
	// create Rec
	_rec.set_directory(o_path);
	_rec.set_period(period);
	_rec.append_tables(_database.tables());
	
}

void Solver::run(){
	clock_t start, end;
	clock_t start0, end0;
	start = clock();
	int stim=-1;
	Gen::mode_t mode=_gen.current_mode();
	int timestamp=_gen.current_time();
	int i=timestamp;
	if(mode!=Gen::END){
		_gen.update();
		mode=_gen.current_mode();
		timestamp=_gen.current_time();
		// Main loop
		while(mode!=Gen::END){
			switch(mode){
			case Gen::RECALL:
				LOG(INFO) << "Sim[" << timestamp << "]:[ RECALL ]";
				start0 = clock();
				_net.recall(timestamp);
				end0 = clock();
				LOG(INFO) << "Time : " << setprecision(6) << (end0-start0)/(double)CLOCKS_PER_SEC;
				_rec.record(timestamp);
				break;
			case Gen::LEARN:
				stim = _gen.current_stim();
				LOG(INFO) << "Sim[" << timestamp << "]:[ LEARN ] with STIM = " << stim;
				start0 = clock();
				_net.learn(timestamp, stim);
				end0 = clock();
				LOG(INFO) << "Time : " << setprecision(6) << (end0-start0)/(double)CLOCKS_PER_SEC;
				_rec.record(timestamp);
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
	_rec.record(timestamp, true);
	end = clock();
	LOG(INFO) << "Total time for [" << timestamp - i -1 << "] steps: "
		<< setprecision(6) << (end-start)/(double)CLOCKS_PER_SEC;
	
	_database.dump_shapes();
	LOG(INFO) << "TABLE HCU_SLOT" << endl << _database.table("hcu_slot")->dump();
}

}
