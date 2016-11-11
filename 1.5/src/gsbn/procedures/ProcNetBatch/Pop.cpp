#include "gsbn/procedures/ProcNetBatch/Pop.hpp"

namespace gsbn{
namespace proc_net_batch{

void Pop::init_new(PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt){
	CHECK(list_pop);
	_list_pop=list_pop;
	
	_id=_list_pop->size();
	list_pop->push_back(this);
	
	_dim_proj = 0;
	_dim_hcu = pop_param.hcu_num();
	_dim_mcu = pop_param.mcu_num();
	CHECK_GT(_dim_hcu, 0);
	CHECK_GT(_dim_mcu, 0);
	_hcu_start = *hcu_cnt;
	_mcu_start = *mcu_cnt;
	*hcu_cnt += _dim_hcu;
	*mcu_cnt += _dim_hcu * _dim_mcu;
	
	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];
	
	_taumdt = dt/pop_param.taum();
	_wtagain = pop_param.wtagain();
	_maxfqdt = pop_param.maxfq()*dt;
	_igain = pop_param.igain();
	_wgain = pop_param.wgain();
	_lgbias = pop_param.lgbias();
	_snoise = pop_param.snoise();
	
	_slot_num = pop_param.slot_num();
	_fanout_num = pop_param.fanout_num();
	
	CHECK(_slot=db.create_sync_vector_i("slot_"+to_string(_id)));
	CHECK(_fanout=db.create_sync_vector_i("fanout_"+to_string(_id)));	
	CHECK(_dsup=db.create_sync_vector_f("dsup_"+to_string(_id)));
	CHECK(_act=db.create_sync_vector_f(".act_"+to_string(_id)));
	CHECK(_epsc=db.create_sync_vector_f("epsc_"+to_string(_id)));
	CHECK(_bj=db.create_sync_vector_f("bj_"+to_string(_id)));
	CHECK(_spike = db.create_sync_vector_i("spike_"+to_string(_id)));
	CHECK(_rnd_uniform01 = db.create_sync_vector_f(".rnd_uniform01"+to_string(_id)));
	CHECK(_rnd_normal = db.create_sync_vector_f(".rnd_normal"+to_string(_id)));
	CHECK(_wmask = db.sync_vector_f(".wmask"));
	CHECK(_lginp = db.sync_vector_f(".lginp"));

	_slot->mutable_cpu_vector()->resize(_dim_hcu, _slot_num);
	_fanout->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, _fanout_num);
	_dsup->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_act->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_spike->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_rnd_uniform01->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_rnd_normal->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	
}

void Pop::init_copy(PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt){
}

void Pop::update_rnd_cpu(){
	float *ptr_uniform01= _rnd_uniform01->mutable_cpu_data();
	float *ptr_normal= _rnd_normal->mutable_cpu_data();
	int size = _dim_hcu * _dim_mcu;
	_rnd.gen_uniform01_cpu(ptr_uniform01, size);
	_rnd.gen_normal_cpu(ptr_normal, size, 0, _snoise);
}

void update_sup_kernel_1_cpu(
	int i,
	int j,
	int dim_proj,
	int dim_hcu,
	int dim_mcu,
	const float *ptr_epsc,
	const float *ptr_bj,
	const float *ptr_lginp,
	const float *ptr_wmask,
	const float *ptr_rnd_normal,
	float *ptr_dsup,
	float wgain,
	float lgbias,
	float igain,
	float taumdt
){
	int idx = i*dim_mcu+j;
	float wsup=0;
	int offset=0;
	int mcu_num_in_pop = dim_proj * dim_hcu * dim_mcu;
	for(int m=0; m<dim_proj; m++){
		wsup += ptr_bj[offset+idx] + ptr_epsc[offset+idx];
		offset += mcu_num_in_pop;
	}
	float sup = lgbias + igain * ptr_lginp[idx] + ptr_rnd_normal[idx];
	sup += (wgain * ptr_wmask[i]) * wsup;
	
	float dsup = ptr_dsup[idx];
	ptr_dsup[idx] += (sup - dsup) * taumdt;
}

void update_sup_kernel_2_cpu(
	int i,
	int dim_mcu,
	const float *ptr_dsup,
	float* ptr_act,
	float wtagain	
){
	float maxdsup = ptr_dsup[0];
	for(int m=0; m<dim_mcu; m++){
		int idx = i*dim_mcu + m;
		float dsup = ptr_dsup[idx];
		if(dsup>maxdsup){
			maxdsup = dsup;
		}
	}
	float maxact = exp(wtagain*maxdsup);
	float vsum = 0;
	for(int m=0; m<dim_mcu; m++){
		int idx = i*dim_mcu+m;
		float dsup = ptr_dsup[idx];
		float act = exp(wtagain*(dsup-maxdsup));
		if(maxact<1){
			act *= maxact;
		}
		vsum += act;
		ptr_act[idx] = act;
	}

	if(vsum>1){
		for(int m=0; m<dim_mcu; m++){
			int idx = i*dim_mcu + m;
			ptr_act[idx] /= vsum;
		}
	}
}

void update_sup_kernel_3_cpu(
	int i,
	int j,
	int dim_mcu,
	const float *ptr_act,
	const float* ptr_rnd_uniform01,
	int* ptr_spk,
	float maxfqdt
){
	int idx = i*dim_mcu+j;
	ptr_spk[idx] = int(ptr_rnd_uniform01[idx]<ptr_act[idx]*maxfqdt);
}

void Pop::update_sup_cpu(){
	const int* ptr_conf = static_cast<const int*>(_conf->cpu_data());
	int lginp_idx = ptr_conf[Database::IDX_CONF_STIM];
	int wmask_idx = ptr_conf[Database::IDX_CONF_GAIN_MASK];
	const float* ptr_wmask = _wmask->cpu_data(wmask_idx)+_hcu_start;
	const float* ptr_lginp = _lginp->cpu_data(lginp_idx)+_mcu_start;
	const float* ptr_epsc = _epsc->cpu_data();
	const float* ptr_bj = _bj->cpu_data();
	const float* ptr_rnd_uniform01 = _rnd_uniform01->cpu_data();
	const float* ptr_rnd_normal = _rnd_normal->cpu_data();
	float* ptr_dsup = _dsup->mutable_cpu_data();
	float* ptr_act = _act->mutable_cpu_data();
	int* ptr_spk = _spike->mutable_cpu_data();
	
	for(int i=0; i<_dim_hcu; i++){
		for(int j=0; j<_dim_mcu; j++){
			update_sup_kernel_1_cpu(
				i,
				j,
				_dim_proj,
				_dim_hcu,
				_dim_mcu,
				ptr_epsc,
				ptr_bj,
				ptr_lginp,
				ptr_wmask,
				ptr_rnd_normal,
				ptr_dsup,
				_wgain,
				_lgbias,
				_igain,
				_taumdt
			);
		}
		update_sup_kernel_2_cpu(
			i,
			_dim_mcu,
			ptr_dsup,
			ptr_act,
			_wtagain
		);
		for(int j=0; j<_dim_mcu; j++){
			update_sup_kernel_3_cpu(
				i,
				j,
				_dim_mcu,
				ptr_act,
				ptr_rnd_uniform01,
				ptr_spk,
				_maxfqdt
			);
		}
	}
}

}
}
