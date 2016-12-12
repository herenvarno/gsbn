#include "gsbn/procedures/ProcRndGen.hpp"

namespace gsbn{
namespace proc_rnd_gen{

REGISTERIMPL(ProcRndGen);

void ProcRndGen::init_new(SolverParam solver_param, Database& db){

	GenParam gen_param = solver_param.gen_param();
	
	CHECK(_conf = db.table(".conf"));
	
	// conf
	float* ptr_conf = static_cast<float*>(_conf->mutable_cpu_data(0));
	ptr_conf[Database::IDX_CONF_DT] = gen_param.dt();
	float dt = gen_param.dt();
	
	// mode
	int mode_param_size = gen_param.mode_param_size();
	int max_step=-1;
	for(int i=0;i<mode_param_size;i++){
		ModeParam mode_param=gen_param.mode_param(i);
		mode_t m;
		
		m.begin_step = int(mode_param.begin_time()/dt);
		m.end_step = int(mode_param.end_time()/dt);
		CHECK_GE(m.begin_step, max_step)
			<< "Order of modes is wrong or there is overlapping time range, abort!";
		CHECK_GE(m.end_step, m.begin_step)
				<< "Time range is wrong, abort!";
		m.prn = mode_param.prn();
		m.lginp_id = 0;
		m.wmask_id = mode_param.gain_mask();
		m.plasticity = mode_param.plasticity();
		_list_mode.push_back(m);
	}
	
	CHECK(_lginp = db.create_sync_vector_f32(".lginp"));
	CHECK(_wmask = db.create_sync_vector_f32(".wmask"));
	
	/*
	string stim_file = gen_param.stim_file();
	StimRawData stim_raw_data;
	fstream input(stim_file, ios::in | ios::binary);
	if (!input) {
		LOG(FATAL) << "File not found!";
	} else if (!stim_raw_data.ParseFromIstream(&input)) {
		LOG(FATAL) << "Parse file error!";
	} else{
		int drows = stim_raw_data.data_rows();
		int dcols = stim_raw_data.data_cols();
		int mrows = stim_raw_data.mask_rows();
		int mcols = stim_raw_data.mask_cols();
		HOST_VECTOR(float, *vdata) = _lginp->mutable_cpu_vector();
		HOST_VECTOR(float, *vmask) = _wmask->mutable_cpu_vector();
		
		_lginp->set_ld(dcols);
		_wmask->set_ld(mcols);
		int data_size = stim_raw_data.data_size();
		for(int i=0; i<data_size; i++){
			vdata->push_back(stim_raw_data.data(i));
		}
		CHECK_EQ(vdata->size(), drows*dcols) << "Bad stimuli!!!";
		
		int mask_size = stim_raw_data.mask_size();
		for(int i=0; i<mask_size; i++){
			vmask->push_back(stim_raw_data.mask(i));
		}
		CHECK_EQ(vmask->size(), mrows*mcols) << "Bad stimuli!!!";
	}
	
	_cursor = 0;
	*/
}

void ProcRndGen::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcRndGen::update_cpu(){
	int _current_step = static_cast<const int *>(_conf->cpu_data())[Database::IDX_CONF_TIMESTAMP];
	_current_step++;
	static_cast<int *>(_conf->mutable_cpu_data())[Database::IDX_CONF_TIMESTAMP]=_current_step;
	
	int r=_list_mode.size();
	for(; _cursor<r; _cursor++){
		mode_t m=_list_mode[_cursor]; 
		if(_current_step > m.begin_step && _current_step <= m.end_step){
			float old_prn = *static_cast<const float *>(_conf->cpu_data(0, Database::IDX_CONF_PRN));
			
			*static_cast<float *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_OLD_PRN))=old_prn;
			*static_cast<float *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_PRN))=m.prn;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_GAIN_MASK))=m.wmask_id;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_PLASTICITY))=m.plasticity;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_STIM))=m.lginp_id;
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_MODE))=1;
			return;
		}else if(_current_step < m.begin_step){
			*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_MODE))=0;
			return;
		}
	}
	*static_cast<int *>(_conf->mutable_cpu_data(0, Database::IDX_CONF_MODE))=-1;
}

#ifndef CPU_ONLY
void ProcRndGen::update_gpu(){
	update_cpu();
}
#endif
}
}
