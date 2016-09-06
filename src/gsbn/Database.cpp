#include "gsbn/Database.hpp"

namespace gsbn{

Database::Database() : _initialized(false), _tables() {

	_tables["proj"] = new Table("proj", {
		sizeof(int),						// SRC_POP
		sizeof(int),						// DESC_POP
		sizeof(int),						// IDX_PROJ_MCU_NUM
		sizeof(float),					// _tauzidt
		sizeof(float),					// _tauzjdt
		sizeof(float),					// _tauedt
		sizeof(float),					// _taupdt
		sizeof(float),					// _eps
		sizeof(float),					// _eps2
		sizeof(float),					// _kfti
		sizeof(float),					// _kftj
		sizeof(float),					// _bgain
		sizeof(float),					// _wgain
		sizeof(float)						// pi0
	});

	_tables["pop"] = new Table("pop", {
		sizeof(int),						// FIRST_HCU_INDEX
		sizeof(int)							// HCU_NUM
	});
	
	_tables["hcu"] = new Table("hcu", {
		sizeof(int),						// FIRST_MCU_INDEX
		sizeof(int),						// MCU_NUM
		sizeof(int),						// IDX_HCU_ISP_INDEX
		sizeof(int),						// IDX_HCU_ISP_NUM
		sizeof(int),						// IDX_HCU_OSP_INDEX
		sizeof(int),						// IDX_HCU_OSP_NUM
		sizeof(float),					// IDX_HCU_TAUMDT
		sizeof(float),					// IDX_HCU_WTAGAIN
		sizeof(float),					// IDX_HCU_MAXFQDT
		sizeof(float),					// IDX_HCU_IGAIN
    sizeof(float),					// IDX_HCU_WGAIN
    sizeof(float)						// IDX_HCU_SNOISE
	});
	
	_tables["hcu_isp"] = new Table("hcu_isp", {
		sizeof(int)							// IDX_HCU_ISP_VALUE
	});
	
	_tables["hcu_osp"] = new Table("hcu_osp", {
		sizeof(int)							// IDX_HCU_OSP_VALUE
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
		sizeof(float),					// Zj
		sizeof(float)						// Bj
	});

	_tables["epsc"] = new Table("epsc", {
		sizeof(float),					// EPSC
	});
	
	_tables["conn"] = new Table("conn", {
		sizeof(int),						// SRC_MUC
		sizeof(int),						// DEST_HCU
		sizeof(int),						// SUBPROJ
		sizeof(int),						// PROJ
		sizeof(int),						// DELAY
		sizeof(int),						// QUEUE
		sizeof(int),						// TYPE
		sizeof(int)							// IJ_MAT_INDEX
	});
	
	_tables["conn0"] = new Table("conn0", {
		sizeof(int),						// SRC_MUC
		sizeof(int),						// DEST_HCU
		sizeof(int),						// SUBPROJ
		sizeof(int),						// PROJ
		sizeof(int),						// DELAY
		sizeof(int),						// QUEUE
		sizeof(int)							// TYPE
	});

	_tables["i_array"] = new Table("i_array", {
		sizeof(float),					// Pi
		sizeof(float),					// Ei
		sizeof(float),					// Zi
		sizeof(float)						// Ti
	});
	
	_tables["ij_mat"] = new Table("ij_mat", {
		sizeof(float),					// Pij
		sizeof(float),					// Eij
		sizeof(float),					// Zi2
		sizeof(float),					// Zj2
		sizeof(float)						// Tij
	});

	_tables["wij"] = new Table("wij", {
		sizeof(float),					// wij
	});
	
	_tables["mode"] = new Table("mode", {
		sizeof(float),					// BEGIN_TIME
		sizeof(float),					// END_TIME
		sizeof(float),					// MODE
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
		sizeof(int),					// IDX_TMP2_SRC_MCU,
		sizeof(int),					// IDX_TMP2_DEST_HCU,
		sizeof(int),					// IDX_TMP2_SUBPROJ
		sizeof(int),					// IDX_TMP2_PROJ,
		sizeof(int)						// IDX_TMP2_IJ_MAT_INDEX
	});
	
	_tables["tmp3"] = new Table("tmp3", {
		sizeof(int),					// IDX_TMP3_CONN,
		sizeof(int),					// IDX_TMP3_DEST_HCU,
		sizeof(int),					// IDX_TMP3_IJ_MAT_IDX,
		sizeof(int),					// IDX_TMP3_PI_INIT,
		sizeof(int)						// IDX_TMP3_PIJ_INIT,
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
		sizeof(float),				// IDX_CONF_TIMESTAMP
		sizeof(float),				// IDX_CONF_DT
		sizeof(float),				// IDX_CONF_PRN
		sizeof(float)					// IDX_CONF_STIM
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
					void *ptr_hcu = static_cast<void *>(_tables["hcu"]->expand(1));
					int *ptr_hcu0 = static_cast<int *>(ptr_hcu);
					float *ptr_hcu1 = static_cast<float *>(ptr_hcu);
					ptr_hcu0[Database::IDX_HCU_MCU_INDEX]=_tables["mcu"]->height();
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
					ptr_hcu0[Database::IDX_HCU_MCU_NUM] = total_mcu_num;
					ptr_hcu1[Database::IDX_HCU_TAUMDT] = dt/hcu_param.taum();
					ptr_hcu1[Database::IDX_HCU_WTAGAIN] = hcu_param.wtagain();
					ptr_hcu1[Database::IDX_HCU_MAXFQDT] = hcu_param.maxfq()*dt;
					ptr_hcu1[Database::IDX_HCU_IGAIN] = hcu_param.igain();
					ptr_hcu1[Database::IDX_HCU_WGAIN] = hcu_param.wgain();
					ptr_hcu1[Database::IDX_HCU_SNOISE] = hcu_param.snoise();
					ptr_hcu1[Database::IDX_HCU_LGBIAS] = hcu_param.lgbias();
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
			void *ptr_proj = static_cast<void *>(_tables["proj"]->expand(1));
			int *ptr_proj0 = static_cast<int *>(ptr_proj);
			float *ptr_proj1 = static_cast<float *>(ptr_proj);
			ptr_proj0[Database::IDX_PROJ_SRC_POP]=src_pop;
			ptr_proj0[Database::IDX_PROJ_DEST_POP]=dest_pop;
			ptr_proj1[Database::IDX_PROJ_TAUZIDT]=dt/proj_param.tauzi();
			ptr_proj1[Database::IDX_PROJ_TAUZJDT]=dt/proj_param.tauzj();
			ptr_proj1[Database::IDX_PROJ_TAUEDT]=dt/proj_param.taue();
			ptr_proj1[Database::IDX_PROJ_TAUPDT]=dt/proj_param.taup();
			ptr_proj1[Database::IDX_PROJ_EPS]=dt/proj_param.taup();
			ptr_proj1[Database::IDX_PROJ_EPS2]=(dt/proj_param.taup())*(dt/proj_param.taup());
			ptr_proj1[Database::IDX_PROJ_KFTI]=1/(proj_param.maxfq() * dt);
			ptr_proj1[Database::IDX_PROJ_KFTJ]=1/(proj_param.maxfq() * dt);
			ptr_proj1[Database::IDX_PROJ_BGAIN]=proj_param.bgain();
			ptr_proj1[Database::IDX_PROJ_WGAIN]=proj_param.wgain();
			ptr_proj1[Database::IDX_PROJ_PI0]=proj_param.pi0();
			int mcu_num=0;
			int hcu_idx=static_cast<const int *>(_tables["pop"]->cpu_data(src_pop))[Database::IDX_POP_HCU_INDEX];
			int hcu_num=static_cast<const int *>(_tables["pop"]->cpu_data(src_pop))[Database::IDX_POP_HCU_NUM];
			for(int j=0; j<hcu_num; j++){
				mcu_num+=static_cast<const int *>(_tables["hcu"]->cpu_data(hcu_idx+j))[Database::IDX_HCU_MCU_NUM];
			}
			ptr_proj0[Database::IDX_PROJ_MCU_NUM]=mcu_num;
		}
	}
	
	// fill mcu, j_array, epsc
	int hcu_num = _tables["hcu"]->height();
	int proj_num = _tables["proj"]->height();
	for(int i=0;i<hcu_num;i++){
		int *ptr_hcu = static_cast<int *>(_tables["hcu"]->mutable_cpu_data(i));
		int mcu_idx = ptr_hcu[Database::IDX_HCU_MCU_INDEX];
		int mcu_num = ptr_hcu[Database::IDX_HCU_MCU_NUM];
		int pop = static_cast<const int *>(_tables["addr"]->cpu_data(mcu_idx))[Database::IDX_ADDR_POP];
		
		int isp_count=0;
		int osp_count=0;
		ptr_hcu[Database::IDX_HCU_ISP_INDEX] = _tables["hcu_isp"]->height();
		ptr_hcu[Database::IDX_HCU_OSP_INDEX] = _tables["hcu_osp"]->height();
		for(int j=0; j<proj_num;j++){
			const int *ptr_proj = static_cast<const int *>(_tables["proj"]->cpu_data(j));
			if(ptr_proj[Database::IDX_PROJ_SRC_POP]==pop){
				int *ptr_hcu_osp=static_cast<int *>(_tables["hcu_osp"]->expand(1));
				ptr_hcu_osp[Database::IDX_HCU_OSP_VALUE]=j;
				osp_count++;
			}
			if(ptr_proj[Database::IDX_PROJ_DEST_POP]==pop){
				LOG(INFO) << "ADD POP" << pop;
				int *ptr_hcu_isp=static_cast<int *>(_tables["hcu_isp"]->expand(1));
				ptr_hcu_isp[Database::IDX_HCU_ISP_VALUE]=j;
				isp_count++;
			}
		}
		ptr_hcu[Database::IDX_HCU_ISP_NUM] = isp_count;
		ptr_hcu[Database::IDX_HCU_OSP_NUM] = osp_count;
		
		for(int j=0; j<mcu_num; j++){
			int *ptr_mcu = static_cast<int *>(_tables["mcu"]->mutable_cpu_data(mcu_idx+j));
			ptr_mcu[Database::IDX_MCU_J_ARRAY_INDEX] = _tables["j_array"]->height();
			ptr_mcu[Database::IDX_MCU_J_ARRAY_NUM] = isp_count;
			for(int k=0; k<isp_count; k++){
				// FIXME: need initialization ???
				float *ptr_epsc = static_cast<float *>(_tables["epsc"]->expand(1));
				float *ptr_j_array = static_cast<float *>(_tables["j_array"]->expand(1));
				ptr_j_array[Database::IDX_J_ARRAY_PJ] = 1.0/mcu_num;
			}
		}
	}
	
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
