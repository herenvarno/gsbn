#include "gsbn/Database.hpp"

namespace gsbn{


Database::Database() : _initialized(false), _tables() {
}

Database::~Database(){
	for(map<string, Table*>::iterator it=_tables.begin(); it!=_tables.end(); it++){
		delete (it->second);
	}
	for(map<string, void*>::iterator it=_sync_vectors.begin(); it!=_sync_vectors.end(); it++){
		if(it->first.substr(it->first.length()-2) == "_i"){
			delete (SyncVector<int>*)(it->second);
		}else if(it->first.substr(it->first.length()-2) == "_f"){
			delete (SyncVector<float>*)(it->second);
		}else{
			delete (SyncVector<double>*)(it->second);
		}
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
	_initialized = true;
}

void Database::init_copy(SolverParam solver_param, SolverState solver_state){
	if(_initialized){
		LOG(WARNING) << "Multiple initializetion of Database detected, ignore!";
		return;
	}
	
	int v_st_size;
	v_st_size = solver_state.vector_state_i_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateI vst = solver_state.vector_state_i(i);
		SyncVector<int> *v;
		CHECK(v = create_sync_vector_i(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_f_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateF vst = solver_state.vector_state_f(i);
		SyncVector<float> *v;
		CHECK(v = create_sync_vector_f(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_d_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateD vst = solver_state.vector_state_d(i);
		SyncVector<double> *v;
		CHECK(v = create_sync_vector_d(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}

	_initialized = true;
}


Table* Database::create_table(const string name, const vector<int> fields){
	if(table(name)){
		LOG(WARNING)<< "table ["<< name << "] already existed!";
		return NULL;
	}
	Table *t = new Table(name, fields);
	register_table(name, t);
	return t;
}

void Database::register_table(const string name, Table *t){
	CHECK(!name.empty());
	CHECK(t);
	
	_tables[name] = t;
}

Table* Database::table(string name){
	map<string, Table*>::iterator it = _tables.find(name);
	if(it!=_tables.end()){
		return it->second;
	}
	return NULL;
}

SyncVector<int>* Database::create_sync_vector_i(const string name){
	if(sync_vector_i(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<int> *v = new SyncVector<int>();
	register_sync_vector_i(name, v);
	return v;
}
SyncVector<float>* Database::create_sync_vector_f(const string name){
	if(sync_vector_f(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<float> *v = new SyncVector<float>();
	register_sync_vector_f(name, v);
	return v;
}
SyncVector<double>* Database::create_sync_vector_d(const string name){
	if(sync_vector_d(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<double> *v = new SyncVector<double>();
	register_sync_vector_d(name, v);
	return v;
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
				vec_st->set_name(it->first.substr(0, it->first.length()-2));
			}else if(it->first.substr(it->first.length()-2) == "_f"){
				VectorStateF *vec_st = st.add_vector_state_f();
				*vec_st=static_cast<SyncVector<float>*>(it->second)->state_f();
				vec_st->set_name(it->first.substr(0, it->first.length()-2));
			}else if(it->first.substr(it->first.length()-2) == "_d"){
				VectorStateD *vec_st = st.add_vector_state_d();
				*vec_st=static_cast<SyncVector<double>*>(it->second)->state_d();
				vec_st->set_name(it->first.substr(0, it->first.length()-2));
			}
		}
	}
	
	return st;
}

}
