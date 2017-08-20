#include "gsbn/procedures/ProcInitConn.hpp"

namespace gsbn{
namespace proc_init_conn{

REGISTERIMPL(ProcInitConn);

void ProcInitConn::init_new(SolverParam solver_param, Database& db){
	NetParam net_param = solver_param.net_param();
	
	GlobalVar _glv;
	
	int rank;
	CHECK(_glv.geti("rank", rank));
	
	ProcParam proc_param = get_proc_param(solver_param);
	
	int pop_id=0;
	int hcu_cnt=0;
	int mcu_cnt=0;
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			Pop p(pop_id, hcu_cnt, mcu_cnt, pop_param, db, rank);
			_pop_list.push_back(p);
		}
	}
	
	int proj_id=0;
	int total_pop_num = pop_id;
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			Prj prj(proj_id, _pop_list, proj_param, db, rank);
			_prj_list.push_back(prj);
		}
	}
	
	Parser par(proc_param);
	string conn_map_file;
	CHECK(par.args("conn-map-file", conn_map_file)) << "Connection map is not specified! Abort";
	
	string line;
	ifstream file_cm(conn_map_file);
	if (file_cm.is_open()){
		while (getline (file_cm,line) ){
			std::stringstream ss(line);
			std::vector<std::string> vstrings;
				
			while(ss.good()){
				string substr;
				getline(ss, substr, ',' );
				vstrings.push_back(substr);
			}
			if(vstrings.size()>2){
				int prj_id = stoi(vstrings[0]);
				Prj prj = _prj_list[prj_id];
				if(prj._rank!=rank){
					continue;
				}
				int hcu_id = stoi(vstrings[1]);
				for(int i=2; i<vstrings.size(); i+=2){
					int ii = stoi(vstrings[i]);
					int di = stoi(vstrings[i+1]);
					add_row(prj_id, ii, hcu_id, di);
				}
			}
		}
		file_cm.close();
	}
	
}

void ProcInitConn::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcInitConn::update_cpu(){
}

#ifndef CPU_ONLY
void ProcInitConn::update_gpu(){
	update_cpu();
}
#endif

void ProcInitConn::add_row(int prj_id, int src_mcu, int dest_hcu, int delay){
	Prj prj = _prj_list[prj_id];
	for(int i=0; i<prj._dim_conn; i++){
		int *ptr_ii = prj._ii->mutable_cpu_data()+dest_hcu*prj._dim_conn;
		int *ptr_di = prj._di->mutable_cpu_data()+dest_hcu*prj._dim_conn;
		if(ptr_ii[i]<0){
			ptr_ii[i]=src_mcu;
			ptr_di[i]=delay;
			break;
		}
	}
}

ProcInitConn::Pop::Pop(int& id, int& hcu_start, int& mcu_start, PopParam pop_param, Database& db, int rank){
	_id = id;
	_rank = pop_param.rank();
	_dim_hcu = pop_param.hcu_num();
	_dim_mcu = pop_param.mcu_num();
	hcu_start += _dim_hcu;
	mcu_start += _dim_hcu * _dim_mcu;
	id++;
}

ProcInitConn::Prj::Prj(int& id, vector<Pop>& pop_list, ProjParam prj_param, Database& db, int rank){
	_id = id;
	_src_pop = prj_param.src_pop();
	_dest_pop = prj_param.dest_pop();
	_dim_hcu = pop_list[_dest_pop]._dim_hcu;
	_dim_mcu = pop_list[_dest_pop]._dim_mcu;
	_dim_conn = pop_list[_src_pop]._dim_hcu * pop_list[_src_pop]._dim_mcu;
	if(_dim_conn > prj_param.slot_num()){
		_dim_conn = prj_param.slot_num();
	}
	_rank = pop_list[_dest_pop]._rank;
	
	// DO NOT CHECK THE RETURN VALUE, SINCE THE VECTORS MAYBE NOT IN THE CURRENT
	// RANK.
	_ii = db.sync_vector_i32("ii_" + to_string(_id));
	_di = db.sync_vector_i32("di_" + to_string(_id));
	
	id++;
}

}
}
