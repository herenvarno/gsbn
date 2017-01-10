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
	v_st_size = solver_state.vector_state_i8_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateI8 vst = solver_state.vector_state_i8(i);
		SyncVector<int8_t> *v;
		CHECK(v = create_sync_vector_i8(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_i16_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateI16 vst = solver_state.vector_state_i16(i);
		SyncVector<int16_t> *v;
		CHECK(v = create_sync_vector_i16(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_i32_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateI32 vst = solver_state.vector_state_i32(i);
		SyncVector<int32_t> *v;
		CHECK(v = create_sync_vector_i32(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_i64_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateI64 vst = solver_state.vector_state_i64(i);
		SyncVector<int64_t> *v;
		CHECK(v = create_sync_vector_i64(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_f16_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateF16 vst = solver_state.vector_state_f16(i);
		SyncVector<fp16> *v;
		CHECK(v = create_sync_vector_f16(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_f32_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateF32 vst = solver_state.vector_state_f32(i);
		SyncVector<float> *v;
		CHECK(v = create_sync_vector_f32(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}
	v_st_size = solver_state.vector_state_f64_size();
	for(int i=0; i<v_st_size; i++){
		VectorStateF64 vst = solver_state.vector_state_f64(i);
		SyncVector<double> *v;
		CHECK(v = create_sync_vector_f64(vst.name()));
		int ds=vst.data_size();
		for(int j=0; j<ds; j++){
			v->mutable_cpu_vector()->push_back(vst.data(j));
		}
		v->set_ld(vst.ld());
	}

	_initialized = true;
}

void Database::set_global_param_i(string key, int32_t val){
	_global_param_i_list[key]=val;
}
void Database::set_global_param_f(string key, float val){
	_global_param_f_list[key]=val;
}
void Database::set_global_param_s(string key, string val){
	_global_param_s_list[key]=val;
}
bool Database::chk_global_param_i(string key){
	if(_global_param_i_list.find(key)==_global_param_i_list.end()){
		return false;
	}
	return true;
}
bool Database::chk_global_param_f(string key){
	if(_global_param_f_list.find(key)==_global_param_f_list.end()){
		return false;
	}
	return true;
}
bool Database::chk_global_param_s(string key){
	if(_global_param_s_list.find(key)==_global_param_s_list.end()){
		return false;
	}
	return true;
}
int Database::get_global_param_i(string key){
	return _global_param_i_list[key];
}
float Database::get_global_param_f(string key){
	return _global_param_f_list[key];
}
string Database::get_global_param_s(string key){
	return _global_param_s_list[key];
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

SyncVector<int8_t>* Database::create_sync_vector_i8(const string name){
	if(sync_vector_i8(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<int8_t> *v = new SyncVector<int8_t>();
	register_sync_vector_i8(name, v);
	return v;
}
SyncVector<int16_t>* Database::create_sync_vector_i16(const string name){
	if(sync_vector_i16(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<int16_t> *v = new SyncVector<int16_t>();
	register_sync_vector_i16(name, v);
	return v;
}
SyncVector<int32_t>* Database::create_sync_vector_i32(const string name){
	if(sync_vector_i32(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<int32_t> *v = new SyncVector<int32_t>();
	register_sync_vector_i32(name, v);
	return v;
}
SyncVector<int64_t>* Database::create_sync_vector_i64(const string name){
	if(sync_vector_i64(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<int64_t> *v = new SyncVector<int64_t>();
	register_sync_vector_i64(name, v);
	return v;
}
SyncVector<fp16>* Database::create_sync_vector_f16(const string name){
	if(sync_vector_f16(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<fp16> *v = new SyncVector<fp16>();
	register_sync_vector_f16(name, v);
	return v;
}
SyncVector<float>* Database::create_sync_vector_f32(const string name){
	if(sync_vector_f32(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<float> *v = new SyncVector<float>();
	register_sync_vector_f32(name, v);
	return v;
}
SyncVector<double>* Database::create_sync_vector_f64(const string name){
	if(sync_vector_f64(name)){
		LOG(WARNING)<< "sync vector ["<< name << "] already existed!";
		return NULL;
	}
	SyncVector<double> *v = new SyncVector<double>();
	register_sync_vector_f64(name, v);
	return v;
}


SyncVector<int8_t>* Database::sync_vector_i8(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_i08");
	if(it!=_sync_vectors.end()){
		return (SyncVector<int8_t>*)(it->second);
	}
	return NULL;
}
SyncVector<int16_t>* Database::sync_vector_i16(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_i16");
	if(it!=_sync_vectors.end()){
		return (SyncVector<int16_t>*)(it->second);
	}
	return NULL;
}
SyncVector<int32_t>* Database::sync_vector_i32(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_i32");
	if(it!=_sync_vectors.end()){
		return (SyncVector<int32_t>*)(it->second);
	}
	return NULL;
}

SyncVector<int64_t>* Database::sync_vector_i64(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_i64");
	if(it!=_sync_vectors.end()){
		return (SyncVector<int64_t>*)(it->second);
	}
	return NULL;
}
SyncVector<fp16>* Database::sync_vector_f16(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_f16");
	if(it!=_sync_vectors.end()){
		return (SyncVector<fp16>*)(it->second);
	}
	return NULL;
}

SyncVector<float>* Database::sync_vector_f32(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_f32");
	if(it!=_sync_vectors.end()){
		return (SyncVector<float>*)(it->second);
	}
	return NULL;
}

SyncVector<double>* Database::sync_vector_f64(const string name){
	map<string, void*>::iterator it = _sync_vectors.find(name+"_f64");
	if(it!=_sync_vectors.end()){
		return (SyncVector<double>*)(it->second);
	}
	return NULL;
}

void Database::register_sync_vector_i8(const string name, SyncVector<int8_t> *v){
	_sync_vectors[name+"_i08"] = (void*)(v);
}
void Database::register_sync_vector_i16(const string name, SyncVector<int16_t> *v){
	_sync_vectors[name+"_i16"] = (void*)(v);
}
void Database::register_sync_vector_i32(const string name, SyncVector<int32_t> *v){
	_sync_vectors[name+"_i32"] = (void*)(v);
}
void Database::register_sync_vector_i64(const string name, SyncVector<int64_t> *v){
	_sync_vectors[name+"_i64"] = (void*)(v);
}
void Database::register_sync_vector_f16(const string name, SyncVector<fp16> *v){
	_sync_vectors[name+"_f16"] = (void*)(v);
}
void Database::register_sync_vector_f32(const string name, SyncVector<float> *v){
	_sync_vectors[name+"_f32"] = (void*)(v);
}
void Database::register_sync_vector_f64(const string name, SyncVector<double> *v){
	_sync_vectors[name+"_f64"] = (void*)(v);
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
			if(it->first.substr(it->first.length()-4) == "_i08"){
				VectorStateI8 *vec_st = st.add_vector_state_i8();
				*vec_st=static_cast<SyncVector<int8_t>*>(it->second)->state_i8();
				vec_st->set_name(it->first.substr(0, it->first.length()-4));
			}else if(it->first.substr(it->first.length()-4) == "_i16"){
				VectorStateI16 *vec_st = st.add_vector_state_i16();
				*vec_st=static_cast<SyncVector<int16_t>*>(it->second)->state_i16();
				vec_st->set_name(it->first.substr(0, it->first.length()-4));
			}else if(it->first.substr(it->first.length()-4) == "_i32"){
				VectorStateI32 *vec_st = st.add_vector_state_i32();
				*vec_st=static_cast<SyncVector<int32_t>*>(it->second)->state_i32();
				vec_st->set_name(it->first.substr(0, it->first.length()-4));
			}else if(it->first.substr(it->first.length()-4) == "_i64"){
				VectorStateI64 *vec_st = st.add_vector_state_i64();
				*vec_st=static_cast<SyncVector<int64_t>*>(it->second)->state_i64();
				vec_st->set_name(it->first.substr(0, it->first.length()-4));
			}else if(it->first.substr(it->first.length()-4) == "_f16"){
				VectorStateF16 *vec_st = st.add_vector_state_f16();
				*vec_st=static_cast<SyncVector<fp16>*>(it->second)->state_f16();
				vec_st->set_name(it->first.substr(0, it->first.length()-4));
			}else if(it->first.substr(it->first.length()-4) == "_f32"){
				VectorStateF32 *vec_st = st.add_vector_state_f32();
				*vec_st=static_cast<SyncVector<float>*>(it->second)->state_f32();
				vec_st->set_name(it->first.substr(0, it->first.length()-4));
			}else if(it->first.substr(it->first.length()-4) == "_f64"){
				VectorStateF64 *vec_st = st.add_vector_state_f64();
				*vec_st=static_cast<SyncVector<double>*>(it->second)->state_f64();
				vec_st->set_name(it->first.substr(0, it->first.length()-4));
			}
		}
	}
	
	return st;
}

}
