#include "gsbn/procedures/ProcExtGen.hpp"

namespace gsbn{
namespace proc_ext_gen{

REGISTERIMPL(ProcExtGen);

void ProcExtGen::init_new(SolverParam solver_param, Database& db){

	GenParam gen_param = solver_param.gen_param();
	_eps = gen_param.eps();
	
	float dt = gen_param.dt();
	_glv.putf("dt", dt);
	
	// mode
	int mode_param_size = gen_param.mode_param_size();
	int max_step=-1;
	for(int i=0;i<mode_param_size;i++){
		ModeParam mode_param=gen_param.mode_param(i);
		
		int begin_step = int(mode_param.begin_time()/dt);
		int end_step = int(mode_param.end_time()/dt);
		CHECK_GE(begin_step, max_step)
			<< "Order of modes is wrong or there is overlapping time range, abort!";
		CHECK_GE(end_step, begin_step)
				<< "Time range is wrong, abort!";
		int begin_lgidx_id = mode_param.begin_lgidx_id();
		int begin_lgexp_id = mode_param.begin_lgexp_id();
		int begin_wmask_id = mode_param.begin_wmask_id();
		int time_step = mode_param.time_step();
		int lgidx_step = mode_param.lgidx_step();
		int lgexp_step = mode_param.lgexp_step();
		int wmask_step = mode_param.wmask_step();
		float prn = mode_param.prn();
		int plasticity = mode_param.plasticity();
		while(begin_step<end_step){
			mode_t m;
			m.begin_step = begin_step;
			m.end_step = begin_step+time_step;
			if(m.end_step>=end_step){
				m.end_step = end_step;
			}
			m.lgidx_id = begin_lgidx_id;
			m.lgexp_id = begin_lgexp_id;
			m.wmask_id = begin_wmask_id;
			begin_lgidx_id+=lgidx_step;
			begin_lgexp_id+=lgexp_step;
			begin_wmask_id+=wmask_step;
			m.prn = prn;
			m.plasticity = plasticity;
			_list_mode.push_back(m);
			begin_step=m.end_step;
		}
	}
	
	CHECK(_lgidx = db.create_sync_vector_i32(".lgidx"));
	CHECK(_wmask = db.create_sync_vector_f32(".wmask"));
	CHECK(_lginp = db.create_sync_vector_f32(".lginp"));
	
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
		HOST_VECTOR(int, *vdata) = _lgidx->mutable_cpu_vector();
		HOST_VECTOR(float, *vmask) = _wmask->mutable_cpu_vector();
		
		_lgidx->set_ld(dcols);
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
	
	
	int mcu_num=0;
	NetParam net_param=solver_param.net_param();
	int pop_param_size = net_param.pop_param_size();
	for(int i=0; i<pop_param_size; i++){
		PopParam pop_param = net_param.pop_param(i);
		mcu_num+=(pop_param.hcu_num()*pop_param.mcu_num());
		for(int j=0; j<pop_param.hcu_num(); j++){
			_mcu_in_hcu.push_back(pop_param.mcu_num());
		}
	}
	
	_lginp->resize(mcu_num);
	
	_cursor = 0;
	_old_lgidx_id =-1;
}

void ProcExtGen::init_copy(SolverParam solver_param, Database& db){
	init_new(solver_param, db);
}

void ProcExtGen::update_cpu(){
	int _current_step;
	CHECK(_glv.geti("simstep", _current_step));
	_current_step++;
	_glv.puti("simstep", _current_step);
	
	int r=_list_mode.size();
	for(; _cursor<r; _cursor++){
		mode_t m=_list_mode[_cursor]; 
		if(_current_step > m.begin_step && _current_step <= m.end_step){
			float old_prn;
			CHECK(_glv.getf("prn", old_prn));
			_glv.putf("old-prn", old_prn);
			_glv.putf("prn", m.prn);
			_glv.puti("wmask-idx", m.wmask_id);
			_glv.putb("plasticity", (bool)(m.plasticity));
			_glv.puti("lginp-idx", 0);	// because the real stimuli are automatically generated.
			_glv.puti("cycle-flag", 1);
			if(_old_lgidx_id != m.lgidx_id){
				_old_lgidx_id = m.lgidx_id;
				const int* ptr_lgidx = _lgidx->cpu_data(m.lgidx_id);
				float* ptr_lginp = _lginp->mutable_cpu_data();
				for(int i=0; i<_mcu_in_hcu.size(); i++){
					for(int j=0; j<_mcu_in_hcu[i]; j++){
						if(*(ptr_lgidx+i)==j){
							*(ptr_lginp)=log(1+_eps);
						}else{
							*(ptr_lginp)=log(0+_eps);
						}
						ptr_lginp++;
					}
				}
			}
			return;
		}else if(_current_step <= m.begin_step){
			_glv.puti("cycle-flag", 0);
			return;
		}
	}
	_glv.puti("cycle-flag", -1);
	
}

#ifndef CPU_ONLY
void ProcExtGen::update_gpu(){
	update_cpu();
}
#endif
}
}
