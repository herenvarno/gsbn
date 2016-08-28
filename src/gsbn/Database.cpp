#include "gsbn/Database.hpp"

namespace gsbn{

Database::Database() : _initialized(false), _tables() {

	_tables["proj"] = new Table("proj", {
		sizeof(int),						// SRC_POP
		sizeof(int)							// DESC_POP
	});

	_tables["pop"] = new Table("pop", {
		sizeof(int),						// FIRST_HCU_INDEX
		sizeof(int)							// HCU_NUM
	});
	
	_tables["hcu"] = new Table("hcu", {
		sizeof(int),						// FIRST_MCU_INDEX
		sizeof(int),						// MCU_NUM
		sizeof(int),						// IDX_HCU_SUBPROJ_INDEX
		sizeof(int)							// IDX_HCU_SUBPROJ_NUM
	});
	
	_tables["hcu_subproj"] = new Table("hcu_subproj", {
		sizeof(int)							// IDX_HCU_SUBPROJ_VALUE
	});
	
	_tables["mcu"] = new Table("mcu", {
		sizeof(int),						// FIRST_J_ARRAY_INDEX
		sizeof(int)							// MCU_PROJ_NUM
	});
	
	_tables["spk"] = new Table("spk", {
		sizeof(unsigned char)		// SPIKE
	});
	
	_tables["mcu_fanout"] = new Table("mcu_fanout", {
		sizeof(int)							// MCU_FANOUT
	});

	_tables["hcu_slot"] = new Table("hcu_slot", {
		sizeof(int)							// HCU_SLOT
	});
	
	_tables["j_array"] = new Table("j_array", {
		sizeof(float),					// Pj
		sizeof(float),					// Ej
		sizeof(float)						// Zj
														// Tj is not needed!
	});

	_tables["epsc"] = new Table("epsc", {
		sizeof(float),					// EPSC
	});
	
	_tables["conn"] = new Table("conn", {
		sizeof(int),						// SRC_MUC
		sizeof(int),						// DEST_HCU
		sizeof(int),						// DEST_MCU
		sizeof(int),						// DEST_SUBPROJ
		sizeof(int),						// DELAY
		sizeof(int),						// QUEUE
		sizeof(int),						// TYPE
		sizeof(int)							// IJ_MAT_INDEX
	});
	
	_tables["conn0"] = new Table("conn0", {
		sizeof(int),						// SRC_MUC
		sizeof(int),						// DEST_HCU
		sizeof(int),						// DEST_MCU
		sizeof(int),						// DEST_SUBPROJ
		sizeof(int),						// DELAY
		sizeof(int),						// QUEUE
		sizeof(int)							// TYPE
	});

	_tables["i_array"] = new Table("i_array", {
		sizeof(float),					// Pi
		sizeof(float),					// Ei
		sizeof(float),					// Zi
		sizeof(int)							// Ti
	});
	
	_tables["ij_mat"] = new Table("ij_mat", {
		sizeof(float),					// Pij
		sizeof(float),					// Eij
		sizeof(float),					// Zi2
		sizeof(float),					// Zj2
		sizeof(int)							// Tij
	});

	_tables["wij"] = new Table("wij", {
		sizeof(float),					// wij
	});
	
	_tables["mode"] = new Table("mode", {
		sizeof(int),						// BEGIN_TIME
		sizeof(int),						// END_TIME
		sizeof(int),						// MODE
		sizeof(int)							// FIRST_STIMULUS_INDEX
	});
	
	_tables["stim"] = new Table("stim", {
		sizeof(float)						// STIMULUS
	});
	
	_tables["tmp1"] = new Table("tmp1", {
		sizeof(int)						// MCU_INDEX
	});
	
	_tables["tmp2"] = new Table("tmp2", {
		sizeof(int),					// IDX_TMP2_CONN,
		sizeof(int),					// IDX_TMP2_DEST_MCU,
		sizeof(int)						// IDX_TMP2_DEST_SUBPROJ,
	});
	
	_tables["tmp3"] = new Table("tmp3", {
		sizeof(int),					// IDX_TMP3_CONN,
		sizeof(int)						// IDX_TMP3_DEST_HCU,
	});
	
	_tables["addr"] = new Table("addr", {
		sizeof(int),					// IDX_ADDR_POP,
		sizeof(int)						// IDX_ADDR_HCU,
	});
	
	_tables["sup"] = new Table("sup", {
		sizeof(float),				// IDX_SUP_DSUP
		sizeof(float)					// IDX_SUP_ACT
	});
	
	_tables["conf"] = new Table("conf", {
		sizeof(float),				// IDX_CONF_KP
		sizeof(float),				// IDX_CONF_KE
		sizeof(float),				// IDX_CONF_KZJ
		sizeof(float),				// IDX_CONF_KZI
		sizeof(float),				// IDX_CONF_KFTJ
		sizeof(float),				// IDX_CONF_KFTI
		sizeof(float),				// IDX_CONF_BGAIN
		sizeof(float),				// IDX_CONF_WGAIN
		sizeof(float),				// IDX_CONF_WTAGAIN
		sizeof(float),				// IDX_CONF_IGAIN
		sizeof(float),				// IDX_CONF_EPS
		sizeof(float),				// IDX_CONF_LGBIAS
		sizeof(float),				// IDX_CONF_SNOISE
		sizeof(float),				// IDX_CONF_MAXFQDT
		sizeof(float)					// IDX_CONF_TAUMDT
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
	
	// Gen
	GenParam gen_param = solver_param.gen_param();
	
	string stim_file = gen_param.stim_file();
	LOG(INFO) << "FILE="<< stim_file;
	StimRawData stim_raw_data;
	fstream input(stim_file, ios::in | ios::binary);
	if (!input) {
		LOG(WARNING) << "File not found!";
	} else if (!stim_raw_data.ParseFromIstream(&input)) {
		LOG(WARNING) << "Parse file error!";
	} else{
		int data_size = stim_raw_data.data_size();
		for(int i=0; i<data_size; i++){
			float *ptr = static_cast<float *>(_tables["stim"]->expand(1));
			if(ptr){
				ptr[Database::IDX_STIM_VALUE] = stim_raw_data.data(i);
			}
		}
	}
	
	int mode_param_size = gen_param.mode_param_size();
	int max_time=-1;
	for(int i=0;i<mode_param_size;i++){
		ModeParam mode_param=gen_param.mode_param(i);
		int *ptr = static_cast<int *>(_tables["mode"]->expand(1));
		if(ptr){
			int begin_time = mode_param.begin_time();
			CHECK_GT(begin_time, max_time)
				<< "Order of modes is wrong or there is overlapping time range, abort!";
			ptr[Database::IDX_MODE_BEGIN_TIME] = begin_time;
			int end_time = mode_param.end_time();
			CHECK_GE(end_time, begin_time)
				<< "Time range is wrong, abort!";
			ptr[Database::IDX_MODE_END_TIME] = end_time;
			max_time = end_time;
			ptr[Database::IDX_MODE_TYPE] = mode_param.type();
			ptr[Database::IDX_MODE_STIM] = mode_param.stim_index();
		}
	}

	// Net
	NetParam net_param = solver_param.net_param();

	// pop, hcu, mcu, hcu_slot, mcu_fanout, spk, sup, addr
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		int pop_num = pop_param.pop_num();
		for(int j=0; j<pop_num; j++){
			int hcu_param_size = pop_param.hcu_param_size();
			int total_hcu_num=0;
			int *ptr_pop = static_cast<int *>(_tables["pop"]->expand(1));
			ptr_pop[Database::IDX_POP_HCU_INDEX]=_tables["hcu"]->height();
			for(int k=0; k<hcu_param_size; k++){
				HcuParam hcu_param = pop_param.hcu_param(k);
				int hcu_num = hcu_param.hcu_num();
				int hcu_slot = hcu_param.slot_num();
				total_hcu_num+=hcu_num;
				for(int l=0;l<hcu_num;l++){
					int mcu_param_size = hcu_param.mcu_param_size();
					int total_mcu_num=0;
					int *ptr_hcu = static_cast<int *>(_tables["hcu"]->expand(1));
					ptr_hcu[Database::IDX_HCU_MCU_INDEX]=_tables["mcu"]->height();
					int *ptr_hcu_slot = static_cast<int *>(_tables["hcu_slot"]->expand(1));
					ptr_hcu_slot[Database::IDX_HCU_SLOT_VALUE]=hcu_slot;
					for(int m=0; m<mcu_param_size; m++){
						McuParam mcu_param = hcu_param.mcu_param(m);
						int mcu_num = mcu_param.mcu_num();
						total_mcu_num += mcu_num;
						int mcu_fanout = mcu_param.fanout_num();
						for(int n=0; n<mcu_num; n++){
							int *ptr_mcu = static_cast<int *>(_tables["mcu"]->expand(1));
							unsigned char *ptr_spk =  static_cast<unsigned char *>(_tables["spk"]->expand(1));
							ptr_spk[Database::IDX_SPK_VALUE]=0;
							int *ptr_mcu_fanout = static_cast<int *>(_tables["mcu_fanout"]->expand(1));
							ptr_mcu_fanout[Database::IDX_MCU_FANOUT_VALUE]=mcu_fanout;
							float *ptr_sup = static_cast<float *>(_tables["sup"]->expand(1));
							ptr_sup[Database::IDX_SUP_DSUP]=0;
							ptr_sup[Database::IDX_SUP_ACT]=0;
							int *ptr_addr = static_cast<int *>(_tables["addr"]->expand(1));
							ptr_addr[Database::IDX_ADDR_POP]=_tables["pop"]->height()-1;
							ptr_addr[Database::IDX_ADDR_HCU]=_tables["hcu"]->height()-1;
						}
					}
					ptr_hcu[Database::IDX_HCU_MCU_NUM]=total_mcu_num;
				}
			}
			ptr_pop[Database::IDX_POP_HCU_NUM]=total_hcu_num;
		}
	}
	
	// proj
	int total_pop_num = _tables["pop"]->height();
	int proj_param_size = net_param.proj_param_size();
	for(int i=0; i<proj_param_size; i++){
		ProjParam proj_param = net_param.proj_param(i);
		int src_pop = proj_param.src_pop();
		int dest_pop = proj_param.dest_pop();
		if(src_pop<total_pop_num && dest_pop<total_pop_num){
			int *ptr_proj = static_cast<int *>(_tables["proj"]->expand(1));
			ptr_proj[Database::IDX_PROJ_SRC_POP]=src_pop;
			ptr_proj[Database::IDX_PROJ_DEST_POP]=dest_pop;
		}
	}
	
	// fill mcu, j_array, epsc
	int hcu_num = _tables["hcu"]->height();
	int proj_num = _tables["proj"]->height();
	for(int i=0;i<hcu_num;i++){
		int mcu_idx = static_cast<const int *>(_tables["hcu"]->cpu_data(i))[Database::IDX_HCU_MCU_INDEX];
		int mcu_num = static_cast<const int *>(_tables["hcu"]->cpu_data(i))[Database::IDX_HCU_MCU_NUM];
		int pop = static_cast<const int *>(_tables["addr"]->cpu_data(mcu_idx))[Database::IDX_ADDR_POP];
		int proj_count=0;
		for(int j=0; j<proj_num;j++){
			const int *ptr_proj = static_cast<const int *>(_tables["proj"]->cpu_data(j));
			if(ptr_proj[Database::IDX_PROJ_SRC_POP]==pop){
				proj_count++;
			}
		}
		if(proj_count > __MAX_SUBPROJ__){	// FIXME
			proj_count = __MAX_SUBPROJ__;
		}
		
		int *ptr_hcu = static_cast<int *>(_tables["hcu"]->mutable_cpu_data(i));
		ptr_hcu[Database::IDX_HCU_SUBPROJ_INDEX] = _tables["hcu_subproj"]->height();
		ptr_hcu[Database::IDX_HCU_SUBPROJ_NUM] = proj_count;
		for(int j=0; j<proj_count; j++){
			int *ptr_hcu_subproj=static_cast<int *>(_tables["hcu_subproj"]->expand(1));
			ptr_hcu_subproj[Database::IDX_HCU_SUBPROJ_VALUE]=-1;
		}
		
		for(int j=0; j<mcu_num; j++){
			int *ptr_mcu = static_cast<int *>(_tables["mcu"]->mutable_cpu_data(mcu_idx+j));
			ptr_mcu[Database::IDX_MCU_J_ARRAY_INDEX] = _tables["j_array"]->height();
			ptr_mcu[Database::IDX_MCU_J_ARRAY_NUM] = proj_count;
			if(proj_count>0){
				_tables["epsc"]->expand(proj_count);	//FIXME : need initialization??
				_tables["j_array"]->expand(proj_count);
			}
		
		}
		
	}
	
	// conf
	ConfParam conf_param = net_param.conf_param();
	float *ptr_conf = static_cast<float*>(_tables["conf"]->expand(1));
	ptr_conf[Database::IDX_CONF_KP] = conf_param.kp();
	ptr_conf[Database::IDX_CONF_KE] = conf_param.ke();
	ptr_conf[Database::IDX_CONF_KZJ] = conf_param.kzj();
	ptr_conf[Database::IDX_CONF_KZI] = conf_param.kzi();
	ptr_conf[Database::IDX_CONF_KFTJ] = conf_param.kftj();
	ptr_conf[Database::IDX_CONF_KFTI] = conf_param.kfti();
	ptr_conf[Database::IDX_CONF_BGAIN] = conf_param.bgain();
	ptr_conf[Database::IDX_CONF_WGAIN] = conf_param.wgain();
	ptr_conf[Database::IDX_CONF_WTAGAIN] = conf_param.wtagain();
	ptr_conf[Database::IDX_CONF_IGAIN] = conf_param.igain();
	ptr_conf[Database::IDX_CONF_EPS] = conf_param.eps();
	ptr_conf[Database::IDX_CONF_LGBIAS] = conf_param.lgbias();
	ptr_conf[Database::IDX_CONF_SNOISE] = conf_param.snoise();
	ptr_conf[Database::IDX_CONF_MAXFQDT] = conf_param.maxfqdt();
	ptr_conf[Database::IDX_CONF_TAUMDT] = conf_param.taumdt();
	
	// nothing to do with tables: tmp1, tmp2, tmp3, conn, conn0, wij
	
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
	dump_shapes();
}


vector<Table*> Database::tables(){
	vector<Table*> ts;
	for(map<string, Table*>::iterator it=_tables.begin(); it!=_tables.end(); it++){
		ts.push_back(it->second);
	}
	return ts;
}

Table* Database::table(string name){
	map<string, Table*>::iterator it = _tables.find(name);
	if(it!=_tables.end()){
		return it->second;
	}
	return NULL;
}



}
