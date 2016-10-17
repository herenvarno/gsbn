#include "gsbn/Database.hpp"

namespace gsbn{


Database::Database() : _initialized(false), _tables() {
	_tables["mode"] = new Table("mode", {
		sizeof(float),					// BEGIN_TIME
		sizeof(float),					// END_TIME
		sizeof(float),					// PRN
		sizeof(int),					// GAIN_MASK
		sizeof(int),						// PLASTICITY
		sizeof(int)							// FIRST_STIMULUS_INDEX
	});
	_tables["conf"] = new Table("conf", {
		sizeof(float),				// IDX_CONF_TIMESTAMP
		sizeof(float),				// IDX_CONF_DT
		sizeof(float),				// IDX_CONF_PRN
		sizeof(int),					// IDX_CONF_GAIN_MASK
		sizeof(int),					// IDX_CONF_PLASTICITY
		sizeof(int)						// IDX_CONF_STIM
	});
}

Database::~Database(){
	for(map<string, Table*>::iterator it=_tables.begin(); it!=_tables.end(); it++){
		delete (it->second);
	}
}



void Database::dump_shapes(){
	for(map<string, Table*>::iterator it=_tables.begin(); it!=_tables.end(); it++){
		LOG(INFO) << (it->second)->name() << " : " << (it->second)->rows() << " x " <<(it->second)->cols() << " = " << (it->second)->height() << " x " << (it->second)->width();
	}
}

void Database::init_new(SolverParam solver_param){
	if(_initialized){
		LOG(WARNING) << "Multiple initializetion of Database detected, ignore!";
		return;
	}
	
	float dt=0.001;
	
	// Gen
	GenParam gen_param = solver_param.gen_param();
	
	string stim_file = gen_param.stim_file();
	StimRawData stim_raw_data;
	fstream input(stim_file, ios::in | ios::binary);
	if (!input) {
		LOG(WARNING) << "File not found!";
	} else if (!stim_raw_data.ParseFromIstream(&input)) {
		LOG(WARNING) << "Parse file error!";
	} else{

	}
	
	// conf
	float *ptr_conf = static_cast<float*>(_tables["conf"]->expand(1));
	ptr_conf[Database::IDX_CONF_DT] = gen_param.dt();
	dt = gen_param.dt();
	
	// mode
	int mode_param_size = gen_param.mode_param_size();
	float max_time=-1;
	for(int i=0;i<mode_param_size;i++){
		ModeParam mode_param=gen_param.mode_param(i);
		float *ptr = static_cast<float *>(_tables["mode"]->expand(1));
		if(ptr){
			float begin_time = mode_param.begin_time();
			CHECK_GT(begin_time, max_time)
				<< "Order of modes is wrong or there is overlapping time range, abort!";
			ptr[Database::IDX_MODE_BEGIN_TIME] = begin_time;
			float end_time = mode_param.end_time();
			CHECK_GE(end_time, begin_time)
				<< "Time range is wrong, abort!";
			ptr[Database::IDX_MODE_END_TIME] = end_time;
			max_time = end_time;
			ptr[Database::IDX_MODE_PRN] = mode_param.prn();
			int *ptr0 = (int *)(ptr);
			ptr0[Database::IDX_MODE_GAIN_MASK] = mode_param.gain_mask();
			ptr0[Database::IDX_MODE_PLASTICITY] = mode_param.plasticity();
			ptr0[Database::IDX_MODE_STIM] = mode_param.stim_index();
		}
	}
	_initialized = true;
	
	dump_shapes();
}

void Database::init_copy(SolverState solver_state){
	if(_initialized){
		LOG(WARNING) << "Multiple initializetion of Database detected, ignore!";
		return;
	}
	int table_state_size = solver_state.table_state_size();
	for(map<string, Table*>::iterator it=_tables.begin(); it!=_tables.end(); it++){
		string name = it->first;
		for(int i=0; i<table_state_size; i++){
			TableState tab_st = solver_state.table_state(i);
			if((it->second)->name() == tab_st.name()){
				(it->second)->set_state(tab_st);
			}
		}
	}
	_initialized = true;
//	dump_shapes();
}


Table* Database::table(string name){
	map<string, Table*>::iterator it = _tables.find(name);
	if(it!=_tables.end()){
		return it->second;
	}
	return NULL;
}

SyncVector<int>* Database::sync_vector_i(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_i");
	if(it!=_sync_vectors.end()){
		return (SyncVector<int>*)(it->second);
	}
	return NULL;
}

SyncVector<float>* Database::sync_vector_f(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_f");
	if(it!=_sync_vectors.end()){
		return (SyncVector<float>*)(it->second);
	}
	return NULL;
}

SyncVector<double>* Database::sync_vector_d(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_d");
	if(it!=_sync_vectors.end()){
		return (SyncVector<double>*)(it->second);
	}
	return NULL;
}

void Database::register_sync_vector_i(const string name, SyncVector<int> *v){
	_sync_vectors[name+"_i"] = (void*)(v);
}
void Database::register_sync_vector_f(const string name, SyncVector<float> *v){
	_sync_vectors[name+"_f"] = (void*)(v);
}
void Database::register_sync_vector_d(const string name, SyncVector<double> *v){
	_sync_vectors[name+"_d"] = (void*)(v);
}

SolverState Database::state_to_proto(){
	SolverState st;
	
	for(map<string, Table*>::iterator it=_tables.begin(); it!=_tables.end(); it++){
		if((it->first)[0]!='.'){
			TableState *tab_st = st.add_table_state();
			*tab_st=it->second->state();
		}
	}
	for(map<string, void*>::iterator it=_sync_vectors.begin(); it!=_sync_vectors.end(); it++){
		if((it->first)[0]!='.'){
			if(it->first.substr(it->first.length()-2) == "_i"){
				VectorStateI *vec_st = st.add_vector_state_i();
				*vec_st=static_cast<SyncVector<int>*>(it->second)->state_i();
				vec_st->set_name(it->first);
			}else if(it->first.substr(it->first.length()-2) == "_f"){
				VectorStateF *vec_st = st.add_vector_state_f();
				*vec_st=static_cast<SyncVector<float>*>(it->second)->state_f();
				vec_st->set_name(it->first);
			}else if(it->first.substr(it->first.length()-2) == "_d"){
				VectorStateD *vec_st = st.add_vector_state_d();
				*vec_st=static_cast<SyncVector<double>*>(it->second)->state_d();
				vec_st->set_name(it->first);
			}
		}
	}
	
	return st;
}

}
